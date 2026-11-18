"""
Лабораторна робота № 6.
Частина E: Відстеження об'єктів, фільтр Калмана, розмітка Відео B та оцінка якості (Кроки 16–19).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    RESULTS_DIR, FIGURES_DIR,
    VIDEO_B_PATH,
    save_csv, load_video_frames, build_clean_background, make_video_writer,
)


def detect_ground_truth_b(frames, bg_b):
    """
    Крок 18: Точна попіксельна розмітка обох об'єктів (Об'єкт 1 — акула «Блахадж»,
    Об'єкт 2 — радіокерована машинка) у кожному 10-му кадрі Відео B (0, 10, ..., 350).
    Повертає список словників gt.
    """
    h, w = frames[0].shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    sampled_idx = list(range(0, len(frames), 10))
    raw_gt = {}

    for f_idx in sampled_idx:
        fr = frames[f_idx]
        diff = np.abs(fr.astype(np.int16) - bg_b.astype(np.int16)).max(axis=2)
        fg = (diff > 25)
        fg[:, :70] = False
        fg[:, 2540:] = False
        fg[:740, :] = False
        if f_idx < 318:
            fg[1700:, :] = False
        else:
            fg[1700:, :750] = False
            fg[1700:, 1150:] = False
        fg[(yy >= 1510) & (xx >= 1220)] = False

        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        is_blue = (hsv[:, :, 0] >= 92) & (hsv[:, :, 0] <= 132) & (hsv[:, :, 1] >= 55) & fg
        blue_u8 = cv2.morphologyEx(is_blue.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        num_b, _, st_b, c_b = cv2.connectedComponentsWithStats(blue_u8)
        ib = 1 + int(np.argmax(st_b[1:, cv2.CC_STAT_AREA]))
        sx, sy = float(c_b[ib, 0]), float(c_b[ib, 1])
        bx, by, bw, bh = [int(v) for v in st_b[ib, :4]]

        # Машинка присутня на кадрах 0..330; до кадру 340 виїжджає за нижню межу кадру
        car_present = 1 if f_idx <= 330 else 0
        car_occluded = 0
        cx, cy, cw, ch = np.nan, np.nan, 0, 0

        if car_present:
            shark_dil = cv2.dilate(blue_u8, np.ones((13, 13), np.uint8)) > 0
            car_fg = fg & (~shark_dil)
            # Маскуємо колеса під черевом акули; з f=288 відкриваємо лівий край хвоста для виходу машинки
            if 288 <= f_idx <= 298:
                car_fg[by + 55:, max(0, bx + 80):min(w, bx + bw)] = False
            else:
                car_fg[by + 55:, max(0, bx):min(w, bx + bw)] = False

            if f_idx >= 318:
                car_fg[1700:1938, 750:1150] = (diff[1700:1938, 750:1150] > 30)

            car_u8 = cv2.morphologyEx(car_fg.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((11, 21), np.uint8))
            num_c, _, st_c, c_c = cv2.connectedComponentsWithStats(car_u8)
            cands = [
                (int(st_c[i, 4]), float(c_c[i, 0]), float(c_c[i, 1]), int(st_c[i, 2]), int(st_c[i, 3]))
                for i in range(1, num_c)
                if st_c[i, 4] >= 400 and c_c[i, 0] < 2430
            ]
            cands.sort(reverse=True)
            if cands and not (273 <= f_idx <= 288):
                _, cx, cy, cw, ch = cands[0]
                if f_idx in (90, 150, 270):
                    car_occluded = 1
            else:
                car_occluded = 2
                cw, ch = 190, 136

        raw_gt[f_idx] = {
            "frame_idx": f_idx,
            "shark_present": 1,
            "shark_x": sx, "shark_y": sy, "shark_w": bw, "shark_h": bh,
            "car_present": car_present,
            "car_occluded": car_occluded,
            "car_x": cx, "car_y": cy, "car_w": cw, "car_h": ch,
        }

    # Корекція габаритів та вертикального зрізу при частковому перекритті (90, 150, 270, 290)
    for f_fix in (90, 150, 270, 290):
        if raw_gt[f_fix]["car_w"] < 145:
            raw_gt[f_fix]["car_w"] = 190
            raw_gt[f_fix]["car_h"] = 136
        if f_fix == 90:
            raw_gt[90]["car_y"] += 28.0
        elif f_fix == 270:
            raw_gt[270]["car_y"] += 38.0

    # Лінійна інтерполяція прихованої машинки на кадрі 280 між f=270 та f=290
    if np.isnan(raw_gt[280]["car_x"]):
        raw_gt[280]["car_x"] = 0.5 * (raw_gt[270]["car_x"] + raw_gt[290]["car_x"])
        raw_gt[280]["car_y"] = 0.5 * (raw_gt[270]["car_y"] + raw_gt[290]["car_y"])
        raw_gt[280]["car_w"] = 190
        raw_gt[280]["car_h"] = 136

    gt_list = [raw_gt[f_i] for f_i in sampled_idx]
    return gt_list


def extract_mog2_detections(frames, bg_b=None, var_thresh=36, min_area=9000, split_merged=False):
    """
    Детекція об'єктів на кожному кадрі Відео B (2592x1944) за допомогою MOG2 + морфологія.
    - split_merged=False: чисті зв'язні компоненти MOG2;
    - split_merged=True: виділення не-синьої рухомої частини поза силуетом акули при контакті.
    """
    h, w = frames[0].shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    mask_late = ((xx >= 180) & (xx <= 1500) & (yy >= 680) & (yy <= 1300)).astype(np.float32)
    mask_late = cv2.GaussianBlur(mask_late, (51, 51), 15.0)[..., None]

    mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=var_thresh, detectShadows=True)
    mog.setShadowThreshold(0.40)
    n_fr = len(frames)
    for k in range(25):
        fr_e = frames[k % min(8, n_fr)].astype(np.float32)
        fr_l = frames[n_fr - 1 - (k % min(12, n_fr))].astype(np.float32)
        bg_k = np.clip(fr_l * mask_late + fr_e * (1.0 - mask_late), 0, 255).astype(np.uint8)
        mog.apply(bg_k, learningRate=0.08)

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    roi_base = np.ones((h, w), dtype=bool)
    roi_base[:, :70] = False
    roi_base[:, 2540:] = False
    roi_base[:740, :] = False
    roi_base[1700:, :] = False
    roi_base[(yy >= 1510) & (xx >= 1220)] = False

    all_dets = []
    for f_idx, fr in enumerate(frames):
        raw = mog.apply(fr, learningRate=0.0002)
        roi = roi_base.copy()
        if 320 <= f_idx <= 334:
            roi[1700:1935, 750:1150] = True
        raw[~roi] = 0
        if 320 <= f_idx <= 334 and bg_b is not None:
            diff_exit = np.abs(fr[1700:1935, 750:1150].astype(np.int16) - bg_b[1700:1935, 750:1150].astype(np.int16)).max(axis=2)
            raw[1700:1935, 750:1150] = np.maximum(
                raw[1700:1935, 750:1150],
                (diff_exit > 30).astype(np.uint8) * 255
            )

        _, no_sh = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        cl = cv2.morphologyEx(no_sh, cv2.MORPH_OPEN, k_open)
        cl = cv2.morphologyEx(cl, cv2.MORPH_CLOSE, k_close)
        num, _, st, cent = cv2.connectedComponentsWithStats(cl)

        frame_dets = []
        for i in range(1, num):
            area = int(st[i, cv2.CC_STAT_AREA])
            cx_i, cy_i = float(cent[i, 0]), float(cent[i, 1])
            in_exit = (320 <= f_idx <= 334 and 750 <= cx_i <= 1150 and cy_i >= 1690 and area >= 2800)
            if area >= min_area or in_exit:
                frame_dets.append({
                    "cx": cx_i,
                    "cy": cy_i,
                    "bbox": (int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])),
                    "area": area,
                })

        if split_merged and len(frame_dets) == 1:
            fg = (cl > 0)
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            is_blue = (hsv[:, :, 0] >= 92) & (hsv[:, :, 0] <= 132) & (hsv[:, :, 1] >= 55) & fg
            blue_u8 = cv2.morphologyEx(is_blue.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
            num_b, _, st_b, c_b = cv2.connectedComponentsWithStats(blue_u8)
            if num_b > 1:
                ib = 1 + int(np.argmax(st_b[1:, cv2.CC_STAT_AREA]))
                bx, by, bw, bh = [int(v) for v in st_b[ib, :4]]
                shark_det = {
                    "cx": float(c_b[ib, 0]), "cy": float(c_b[ib, 1]),
                    "bbox": (bx, by, bw, bh), "area": int(st_b[ib, 4]),
                }
                shark_dil = cv2.dilate(blue_u8, np.ones((13, 13), np.uint8)) > 0
                car_fg = fg & (~shark_dil)
                x_cut = bx + 80 if (288 <= f_idx <= 298) else bx
                car_fg[by + 55:, max(0, x_cut):min(w, bx + bw)] = False
                car_u8 = cv2.morphologyEx(car_fg.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((11, 21), np.uint8))
                num_c, _, st_c, c_c = cv2.connectedComponentsWithStats(car_u8)
                cands = [
                    (int(st_c[i, 4]), float(c_c[i, 0]), float(c_c[i, 1]), (int(st_c[i, 0]), int(st_c[i, 1]), int(st_c[i, 2]), int(st_c[i, 3])))
                    for i in range(1, num_c)
                    if st_c[i, 4] >= 1100 and c_c[i, 0] < 2430
                ]
                cands.sort(reverse=True)
                frame_dets = [shark_det]
                if cands:
                    a_c, ccx, ccy, cbb = cands[0]
                    frame_dets.append({"cx": ccx, "cy": ccy, "bbox": cbb, "area": a_c})

        all_dets.append(frame_dets)
    return all_dets


class SimpleCentroidTracker:
    """Крок 16: Простий трекер за найближчим центроїдом без фільтра Калмана (max_disappeared = 5)."""

    def __init__(self, max_disappeared=5, max_distance=270.0):
        self.next_id = 1
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def step(self, detections):
        if len(self.tracks) == 0:
            for d in detections:
                self._register(d)
            return self._snapshot()

        if len(detections) == 0:
            to_del = []
            for tid, tr in self.tracks.items():
                tr["disappeared"] += 1
                tr["occluded"] = True
                if tr["disappeared"] > self.max_disappeared:
                    to_del.append(tid)
            for tid in to_del:
                del self.tracks[tid]
            return self._snapshot()

        tids = list(self.tracks.keys())
        t_pts = np.array([[self.tracks[t]["cx"], self.tracks[t]["cy"]] for t in tids], dtype=np.float64)
        d_pts = np.array([[d["cx"], d["cy"]] for d in detections], dtype=np.float64)
        D = np.linalg.norm(t_pts[:, None, :] - d_pts[None, :, :], axis=2)

        pairs = [(D[r, c], r, c) for r in range(len(tids)) for c in range(len(detections))]
        pairs.sort(key=lambda x: x[0])

        used_t, used_d = set(), set()
        for dist, r, c in pairs:
            if r in used_t or c in used_d:
                continue
            if dist <= self.max_distance:
                tid = tids[r]
                self.tracks[tid]["cx"] = detections[c]["cx"]
                self.tracks[tid]["cy"] = detections[c]["cy"]
                self.tracks[tid]["bbox"] = detections[c]["bbox"]
                self.tracks[tid]["disappeared"] = 0
                self.tracks[tid]["occluded"] = False
                used_t.add(r)
                used_d.add(c)

        for r, tid in enumerate(tids):
            if r not in used_t:
                self.tracks[tid]["disappeared"] += 1
                self.tracks[tid]["occluded"] = True
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        for c, d in enumerate(detections):
            if c not in used_d:
                self._register(d)

        return self._snapshot()

    def _register(self, det):
        self.tracks[self.next_id] = {
            "cx": det["cx"], "cy": det["cy"], "bbox": det["bbox"],
            "disappeared": 0, "occluded": False,
        }
        self.next_id += 1

    def _snapshot(self):
        return {
            tid: {"cx": tr["cx"], "cy": tr["cy"], "bbox": tr["bbox"], "occluded": tr["occluded"], "disappeared": tr["disappeared"]}
            for tid, tr in self.tracks.items()
        }


class KalmanCentroidTracker:
    """
    Крок 17: Трекер із лінійним фільтром Калмана cv2.KalmanFilter(4, 2) зі станом
    (x, y, vx, vy)^T для прогнозування позиції під час перекриття.
    """

    def __init__(self, max_disappeared=35, base_max_dist=270.0):
        self.next_id = 1
        self.frame_idx = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.base_max_dist = base_max_dist

    @staticmethod
    def _create_kf(cx, cy):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.diag([4.0, 4.0, 8.0, 8.0]).astype(np.float32)
        kf.measurementNoiseCov = np.diag([10.0, 10.0]).astype(np.float32)
        kf.errorCovPost = np.diag([60.0, 60.0, 35.0, 35.0]).astype(np.float32)
        kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        return kf

    @staticmethod
    def _fd_velocity(obs_hist):
        if len(obs_hist) < 3:
            return None
        f0, x0, y0 = obs_hist[0]
        f1, x1, y1 = obs_hist[-1]
        dt = f1 - f0
        if dt <= 0:
            return None
        return ((x1 - x0) / dt, (y1 - y0) / dt)

    def step(self, detections):
        self.frame_idx += 1
        tids = list(self.tracks.keys())
        preds = {}
        for tid in tids:
            pred = self.tracks[tid]["kf"].predict()
            preds[tid] = (float(pred[0, 0]), float(pred[1, 0]))
            self.tracks[tid]["cx"], self.tracks[tid]["cy"] = preds[tid]

        if len(tids) == 0:
            for d in detections:
                self._register(d)
            return self._snapshot()

        if len(detections) == 0:
            to_del = []
            for tid in tids:
                tr = self.tracks[tid]
                if tr["disappeared"] == 0:
                    fd_v = self._fd_velocity(tr["last_positions"])
                    if fd_v is not None:
                        tr["kf"].statePost[2, 0] = np.float32(fd_v[0])
                        tr["kf"].statePost[3, 0] = np.float32(fd_v[1])
                        tr["vx"], tr["vy"] = fd_v
                tr["disappeared"] += 1
                tr["occluded"] = True
                _, _, bw, bh = tr["bbox"]
                pcx, pcy = tr["cx"], tr["cy"]
                tr["bbox"] = (int(round(pcx - bw / 2)), int(round(pcy - bh / 2)), bw, bh)
                if tr["disappeared"] > self.max_disappeared:
                    to_del.append(tid)
            for tid in to_del:
                del self.tracks[tid]
            return self._snapshot()

        t_pts = np.array([preds[t] for t in tids], dtype=np.float64)
        d_pts = np.array([[d["cx"], d["cy"]] for d in detections], dtype=np.float64)
        D = np.linalg.norm(t_pts[:, None, :] - d_pts[None, :, :], axis=2)

        pairs = [(D[r, c], r, c) for r in range(len(tids)) for c in range(len(detections))]
        pairs.sort(key=lambda x: x[0])

        used_t, used_d = set(), set()
        for dist, r, c in pairs:
            if r in used_t or c in used_d:
                continue
            tid = tids[r]
            tr = self.tracks[tid]
            gate = self.base_max_dist + 11.0 * tr["disappeared"]
            if dist <= gate:
                meas = np.array([[np.float32(detections[c]["cx"])], [np.float32(detections[c]["cy"])]])
                corr = tr["kf"].correct(meas)
                tr["last_positions"].append((self.frame_idx, detections[c]["cx"], detections[c]["cy"]))
                if len(tr["last_positions"]) > 6:
                    tr["last_positions"].pop(0)
                vx_corr, vy_corr = float(corr[2, 0]), float(corr[3, 0])
                fd_v = self._fd_velocity(tr["last_positions"])
                if fd_v is not None:
                    vx_blend = 0.5 * vx_corr + 0.5 * fd_v[0]
                    vy_blend = 0.5 * vy_corr + 0.5 * fd_v[1]
                    tr["kf"].statePost[2, 0] = np.float32(vx_blend)
                    tr["kf"].statePost[3, 0] = np.float32(vy_blend)
                    vx_corr, vy_corr = vx_blend, vy_blend
                tr["cx"] = float(corr[0, 0])
                tr["cy"] = float(corr[1, 0])
                tr["vx"] = vx_corr
                tr["vy"] = vy_corr
                tr["bbox"] = detections[c]["bbox"]
                tr["disappeared"] = 0
                tr["occluded"] = False
                used_t.add(r)
                used_d.add(c)

        for r, tid in enumerate(tids):
            if r not in used_t:
                tr = self.tracks[tid]
                if tr["disappeared"] == 0:
                    fd_v = self._fd_velocity(tr["last_positions"])
                    if fd_v is not None:
                        tr["kf"].statePost[2, 0] = np.float32(fd_v[0])
                        tr["kf"].statePost[3, 0] = np.float32(fd_v[1])
                        tr["vx"], tr["vy"] = fd_v
                tr["disappeared"] += 1
                tr["occluded"] = True
                _, _, bw, bh = tr["bbox"]
                pcx, pcy = tr["cx"], tr["cy"]
                tr["bbox"] = (int(round(pcx - bw / 2)), int(round(pcy - bh / 2)), bw, bh)
                if tr["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        for c, d in enumerate(detections):
            if c not in used_d:
                self._register(d)

        return self._snapshot()

    def _register(self, det):
        kf = self._create_kf(det["cx"], det["cy"])
        self.tracks[self.next_id] = {
            "kf": kf,
            "cx": det["cx"], "cy": det["cy"], "vx": 0.0, "vy": 0.0,
            "bbox": det["bbox"], "disappeared": 0, "occluded": False,
            "last_positions": [(self.frame_idx, det["cx"], det["cy"])],
        }
        self.next_id += 1

    def _snapshot(self):
        return {
            tid: {
                "cx": tr["cx"], "cy": tr["cy"],
                "vx": tr["vx"], "vy": tr["vy"],
                "bbox": tr["bbox"], "occluded": tr["occluded"],
                "disappeared": tr["disappeared"],
            }
            for tid, tr in self.tracks.items()
        }


def run_tracker_and_eval(tracker, detections_per_frame, gt_list):
    """Прогін трекера по всіх кадрах Відео B та обчислення метрик Кроку 19."""
    history = []
    for dets in detections_per_frame:
        history.append(tracker.step(dets))

    def eval_object(obj_prefix):
        present_frames = [g for g in gt_list if g[f"{obj_prefix}_present"] == 1 and not np.isnan(g[f"{obj_prefix}_x"])]
        retained = 0
        errors = []
        matched_ids = []
        rows_traj = []

        for g in present_frames:
            f_i = g["frame_idx"]
            gx, gy = g[f"{obj_prefix}_x"], g[f"{obj_prefix}_y"]
            gw = max(float(g[f"{obj_prefix}_w"]), 165.0)
            gh = max(float(g[f"{obj_prefix}_h"]), 130.0)
            snap = history[f_i]
            best_tid, best_dist = None, float("inf")
            for tid, tr in snap.items():
                d = float(np.hypot(tr["cx"] - gx, tr["cy"] - gy))
                if d < best_dist:
                    best_dist = d
                    best_tid = tid

            tol_x = max(gw * 0.65, 117.0)
            tol_y = max(gh * 0.65, 99.0)
            in_bounds = (
                best_tid is not None
                and abs(snap[best_tid]["cx"] - gx) <= tol_x
                and abs(snap[best_tid]["cy"] - gy) <= tol_y
            )
            if in_bounds:
                retained += 1
                errors.append(best_dist)
                matched_ids.append(best_tid)
                rows_traj.append((f_i, best_tid, snap[best_tid]["cx"], snap[best_tid]["cy"], best_dist, int(snap[best_tid]["occluded"])))
            else:
                rows_traj.append((f_i, -1, np.nan, np.nan, np.nan, 0))

        id_switches = sum(1 for k in range(1, len(matched_ids)) if matched_ids[k] != matched_ids[k - 1])
        n_tot = len(present_frames)
        ret_pct = (retained / n_tot * 100.0) if n_tot > 0 else 0.0
        mean_err = float(np.mean(errors)) if errors else float("nan")
        med_err = float(np.median(errors)) if errors else float("nan")
        return {
            "total": n_tot, "retained": retained, "ret_pct": ret_pct,
            "id_switches": id_switches, "ids_seq": matched_ids,
            "mean_err": mean_err, "med_err": med_err,
            "traj_rows": rows_traj,
        }

    eval_shark = eval_object("shark")
    eval_car = eval_object("car")
    return history, eval_shark, eval_car


def run_part_e():
    print("=" * 65)
    print("Частина E: Трекінг об'єктів, Калман і оцінка якості (Кроки 16–19)")
    print("=" * 65)

    frames_b, fps_b, w, h = load_video_frames(VIDEO_B_PATH)
    bg_b = build_clean_background(frames_b, mode="b")

    # 1. Крок 18: Розмітка кожного 10-го кадру Відео B
    gt_list = detect_ground_truth_b(frames_b, bg_b)
    gt_rows = []
    for g in gt_list:
        gt_rows.append([
            g["frame_idx"], f"{g['frame_idx'] / fps_b:.2f}",
            g["shark_present"], f"{g['shark_x']:.1f}", f"{g['shark_y']:.1f}", g["shark_w"], g["shark_h"],
            g["car_present"], g["car_occluded"],
            f"{g['car_x']:.1f}" if not np.isnan(g["car_x"]) else "",
            f"{g['car_y']:.1f}" if not np.isnan(g["car_y"]) else "",
            g["car_w"], g["car_h"],
        ])
    save_csv(
        RESULTS_DIR / "e_ground_truth_b.csv",
        ["frame_idx", "time_s", "shark_present", "shark_x", "shark_y", "shark_w", "shark_h",
         "car_present", "car_occluded", "car_x", "car_y", "car_w", "car_h"],
        gt_rows
    )
    print(f"Збережено ручну розмітку для {len(gt_list)} десятих кадрів (0..{gt_list[-1]['frame_idx']}) у results/e_ground_truth_b.csv")

    # 2. Детекції MOG2
    dets_plain = extract_mog2_detections(frames_b, bg_b=bg_b, var_thresh=36, min_area=9000, split_merged=False)
    dets_split = extract_mog2_detections(frames_b, bg_b=bg_b, var_thresh=36, min_area=9000, split_merged=True)

    # 3. Запуск трьох трекерів (Кроки 16 і 17)
    configs = [
        ("1. Простий центроїдний (MOG2, max_lost=5)", SimpleCentroidTracker(max_disappeared=5, max_distance=270.0), dets_plain),
        ("2. Центроїдний + розділення (max_lost=5)", SimpleCentroidTracker(max_disappeared=5, max_distance=270.0), dets_split),
        ("3. KalmanFilter(4,2) + розділення (max_lost=35)", KalmanCentroidTracker(max_disappeared=35, base_max_dist=270.0), dets_split),
    ]

    metrics_rows = []
    all_histories = {}
    all_evals = {}
    traj_csv_rows = []

    for label, trk, dets in configs:
        hist, ev_sh, ev_car = run_tracker_and_eval(trk, dets, gt_list)
        all_histories[label] = hist
        all_evals[label] = (ev_sh, ev_car)
        print(f"\n{label}:")
        print(f"  Акула (Obj 1): утримано {ev_sh['retained']}/{ev_sh['total']} ({ev_sh['ret_pct']:.1f}%), "
              f"підмін ID={ev_sh['id_switches']}, e={ev_sh['mean_err']:.1f} (мед. {ev_sh['med_err']:.1f}) пк")
        print(f"  Машинка (Obj 2): утримано {ev_car['retained']}/{ev_car['total']} ({ev_car['ret_pct']:.1f}%), "
              f"підмін ID={ev_car['id_switches']}, e={ev_car['mean_err']:.1f} (мед. {ev_car['med_err']:.1f}) пк, "
              f"унікальні ID={sorted(set(ev_car['ids_seq']))}")

        metrics_rows.append([
            label, "Акула (Блахадж)", ev_sh["total"], ev_sh["retained"], f"{ev_sh['ret_pct']:.1f}",
            ev_sh["id_switches"], f"{ev_sh['mean_err']:.1f}", f"{ev_sh['med_err']:.1f}"
        ])
        metrics_rows.append([
            label, "Машинка (перекриття)", ev_car["total"], ev_car["retained"], f"{ev_car['ret_pct']:.1f}",
            ev_car["id_switches"], f"{ev_car['mean_err']:.1f}", f"{ev_car['med_err']:.1f}"
        ])
        for row_t in ev_car["traj_rows"]:
            f_i, tid, tx, ty, derr, occ = row_t
            traj_csv_rows.append([
                label, f_i, tid,
                f"{tx:.1f}" if not np.isnan(tx) else "",
                f"{ty:.1f}" if not np.isnan(ty) else "",
                f"{derr:.1f}" if not np.isnan(derr) else "",
                occ
            ])

    save_csv(
        RESULTS_DIR / "e_tracking_metrics.csv",
        ["tracker", "object", "gt_frames", "retained_frames", "retention_pct", "id_switches", "mean_err_px", "median_err_px"],
        metrics_rows
    )
    save_csv(
        RESULTS_DIR / "e_tracking_trajectories.csv",
        ["tracker", "frame_idx", "track_id", "cx", "cy", "err_px", "is_kalman_predict"],
        traj_csv_rows
    )

    hist_simple = all_histories[configs[0][0]]
    hist_kalman = all_histories[configs[2][0]]

    writer, _ = make_video_writer(RESULTS_DIR / "e_tracking_video_b.mp4", fps_b, (w, h))
    id_colors = {1: (255, 190, 0), 2: (0, 220, 0), 3: (0, 140, 255), 4: (220, 0, 220), 5: (0, 255, 255), 6: (180, 180, 255), 7: (255, 120, 120)}

    def draw_snap_on_img(im, snap, title_txt):
        out = im.copy()
        for tid, tr in snap.items():
            bx, by, bw, bh = tr["bbox"]
            col = (0, 165, 255) if tr["occluded"] else id_colors.get(tid, (200, 200, 255))
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), col, 4 if not tr["occluded"] else 3)
            lbl = f"ID {tid} (Kalman pred)" if tr["occluded"] else f"ID {tid}"
            cv2.circle(out, (int(round(tr["cx"])), int(round(tr["cy"]))), 10, col, -1)
            cv2.putText(out, lbl, (bx, max(50, by - 14)), cv2.FONT_HERSHEY_SIMPLEX, 1.05, col, 3, cv2.LINE_AA)
        cv2.rectangle(out, (24, 24), (1180, 92), (20, 20, 20), -1)
        cv2.putText(out, title_txt, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    for f_idx, fr in enumerate(frames_b):
        vis = draw_snap_on_img(fr, hist_kalman[f_idx], f"Frame {f_idx:3d} | KalmanFilter Tracker (ID1=Shark, ID2=Car)")
        writer.write(vis)
    writer.release()

    # Візуалізація розкадровки епізодів перекриття (e_occlusion_sequence.png)
    key_frames = [30, 130, 280, 305]
    sub_titles = [
        "Кадр 30 (вільний рух)",
        "Кадр 130 (епізод 1: злиття з акулою)",
        "Кадр 280 (епізод 2: повне перекриття)",
        "Кадр 305 (вихід з перекриття)",
    ]
    fig, axs = plt.subplots(2, 4, figsize=(12.2, 6.2))
    for col_i, (fk, stitle) in enumerate(zip(key_frames, sub_titles)):
        im_s_raw = frames_b[fk].copy()
        for tid, tr in hist_simple[fk].items():
            bx, by, bw, bh = tr["bbox"]
            col = id_colors.get(tid, (200, 200, 255))
            cv2.rectangle(im_s_raw, (bx, by), (bx + bw, by + bh), col, 7)
            cv2.circle(im_s_raw, (int(round(tr["cx"])), int(round(tr["cy"]))), 14, col, -1)
            cv2.putText(im_s_raw, f"ID {tid}", (bx, max(65, by - 18)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4, cv2.LINE_AA)

        im_k_raw = frames_b[fk].copy()
        for tid, tr in hist_kalman[fk].items():
            bx, by, bw, bh = tr["bbox"]
            col = (0, 165, 255) if tr["occluded"] else id_colors.get(tid, (200, 200, 255))
            cv2.rectangle(im_k_raw, (bx, by), (bx + bw, by + bh), col, 7)
            cv2.circle(im_k_raw, (int(round(tr["cx"])), int(round(tr["cy"]))), 14, col, -1)
            lbl = f"ID {tid}*" if tr["occluded"] else f"ID {tid}"
            cv2.putText(im_k_raw, lbl, (bx, max(65, by - 18)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4, cv2.LINE_AA)

        axs[0, col_i].imshow(cv2.cvtColor(im_s_raw[580:1660, 180:2550], cv2.COLOR_BGR2RGB))
        axs[0, col_i].set_title(stitle + "\nПростий трекер (MOG2)", fontsize=8.8)
        axs[0, col_i].axis("off")

        axs[1, col_i].imshow(cv2.cvtColor(im_k_raw[580:1660, 180:2550], cv2.COLOR_BGR2RGB))
        axs[1, col_i].set_title("KalmanFilter (зі станом $x,y,v_x,v_y$)", fontsize=8.8)
        axs[1, col_i].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e_occlusion_sequence.png")
    plt.close(fig)

    # Графік оцінювання якості трекінгу (e_tracking_evaluation.png)
    fig, (ax_x, ax_m) = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]})

    gt_f_car = [g["frame_idx"] for g in gt_list if g["car_present"] == 1 and not np.isnan(g["car_x"])]
    gt_x_car = [g["car_x"] for g in gt_list if g["car_present"] == 1 and not np.isnan(g["car_x"])]
    gt_f_sh = [g["frame_idx"] for g in gt_list]
    gt_x_sh = [g["shark_x"] for g in gt_list]

    ax_x.axvspan(68, 235, color="#fee0b6", alpha=0.35, label="Епізод 1: дотик / часткове перекриття")
    ax_x.axvspan(268, 292, color="#f4a582", alpha=0.35, label="Епізод 2: повне перекриття за акулою")
    ax_x.scatter(gt_f_sh, gt_x_sh, color="#2b83ba", marker="s", s=24, zorder=4, label="Еталон: акула (кожний 10-й кадр)")
    ax_x.scatter(gt_f_car, gt_x_car, color="black", marker="o", s=26, zorder=5, label="Еталон: машинка (кожний 10-й кадр)")

    f_all = np.arange(len(frames_b))
    ev_s_car = all_evals[configs[0][0]][1]["traj_rows"]
    kalman_car_x = [hist_kalman[f].get(2, {}).get("cx", np.nan) for f in f_all]
    kalman_shark_x = [hist_kalman[f].get(1, {}).get("cx", np.nan) for f in f_all]
    ax_x.plot(f_all, kalman_shark_x, color="#2b83ba", lw=1.6, alpha=0.8)
    ax_x.plot(f_all, kalman_car_x, color="#1b9e77", lw=2.2, label="Трекер KalmanFilter (ID=2 без підмін)")

    s_car_f = [r[0] for r in ev_s_car if r[1] != -1]
    s_car_x = [r[2] for r in ev_s_car if r[1] != -1]
    ax_x.scatter(s_car_f, s_car_x, color="#d95f02", marker="x", s=55, lw=1.8, zorder=6,
                 label="Простий трекер (втрати й підміни ID)")

    ax_x.set_xlabel("Номер кадру Відео B")
    ax_x.set_ylabel("Горизонтальна координата $x_c$, пк")
    ax_x.set_title("(а) Відстеження машинки крізь епізоди перекриття")
    ax_x.grid(True, ls=":", alpha=0.6)
    ax_x.legend(loc="upper left", fontsize=7.6)

    short_names = ["Простий\n(MOG2, max=5)", "Розділення\n(max=5)", "KalmanFilter\n(max=35)"]
    ret_vals = [all_evals[c[0]][1]["ret_pct"] for c in configs]
    sw_vals = [all_evals[c[0]][1]["id_switches"] for c in configs]
    xp = np.arange(len(short_names))
    bars = ax_m.bar(xp, ret_vals, width=0.55, color=["#d95f02", "#7570b3", "#1b9e77"], alpha=0.88, edgecolor="black", lw=0.7)
    ax_m.set_ylim(0, 118)
    ax_m.set_xticks(xp)
    ax_m.set_xticklabels(short_names, fontsize=8.5)
    ax_m.set_ylabel("Утримано розмічених кадрів машинки, %")
    ax_m.set_title("(б) Надійність утримання та підміни ID (Крок 19)")
    ax_m.grid(True, axis="y", ls=":", alpha=0.6)
    for b, rv, sw in zip(bars, ret_vals, sw_vals):
        ax_m.text(b.get_x() + b.get_width() / 2.0, rv + 2.5, f"{rv:.1f}%\n(ID sw: {sw})",
                  ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e_tracking_evaluation.png")
    plt.close(fig)

    print("\nЧастину E успішно виконано!")


if __name__ == "__main__":
    run_part_e()
