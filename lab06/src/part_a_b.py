"""
Лабораторна робота № 6.
Частина A: Перевірка відеопослідовностей та стабільності експозиції (Кроки 1–2).
Частина B: Розріджений оптичний потік Лукаса–Канаде та зворотна перевірка (Кроки 3–7).
"""

from collections import deque
import gc
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    RESULTS_DIR, FIGURES_DIR,
    VIDEO_A_PATH, VIDEO_B_PATH,
    save_csv, load_video_frames, build_clean_background,
    active_roi_mask, make_video_writer,
)


def analyze_exposure(frames, fps, label, file_name):
    """Крок 2: Обчислення середньої яскравості кадрів для перевірки стабільності експозиції."""
    means = []
    for fr in frames:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        means.append(float(gray.mean()))

    ma = np.array(means, dtype=np.float64)
    h, w = frames[0].shape[:2]
    n = len(frames)
    row = [
        label,
        file_name,
        f"{w}x{h}",
        f"{fps:.2f}",
        n,
        f"{n / fps:.2f}",
        f"{ma.mean():.2f}",
        f"{ma.std():.2f}",
        f"{ma.min():.2f}",
        f"{ma.max():.2f}",
        f"{ma.max() - ma.min():.2f}",
    ]
    return row, ma


def run_sparse_lk(frames, bg_img, video_tag, out_video_path, fps, snap_frames):
    """
    Кроки 3–7: Виділення кутів Ши–Томасі (maxCorners=300, qualityLevel=0.01, minDistance=10),
    пірамідальний потік Лукаса–Канаде (21x21, 3 рівні) без фільтрації та зі зворотною
    перевіркою (поріг 1.0 пк), накопичення 20-кадрових траєкторій та запис відео.
    """
    h, w = frames[0].shape[:2]
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    roi_mask = active_roi_mask(h, w, margin=10)

    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=300, qualityLevel=0.01, minDistance=10, mask=roi_mask)
    if p0 is None:
        raise RuntimeError("Не знайдено жодної початкової точки goodFeaturesToTrack.")
    n_init = len(p0)

    # Класифікація початкових точок на точки рухомих об'єктів (великі зв'язні компоненти >= 4000 пк) та статичного фону
    diff0 = np.abs(frames[0].astype(np.int16) - bg_img.astype(np.int16)).max(axis=2)
    fg_raw = cv2.morphologyEx((diff0 > 28).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    num_fg, lbl_fg, st_fg, _ = cv2.connectedComponentsWithStats(fg_raw)
    fg_clean = np.zeros_like(fg_raw)
    for ci in range(1, num_fg):
        if st_fg[ci, cv2.CC_STAT_AREA] >= 4000:
            fg_clean[lbl_fg == ci] = 255
    fg0 = cv2.dilate(fg_clean, np.ones((11, 11), np.uint8))
    is_obj = np.array([fg0[int(round(y)), int(round(x))] > 0 for x, y in p0.reshape(-1, 2)], dtype=bool)
    n_init_obj = int(is_obj.sum())
    n_init_bg = int((~is_obj).sum())

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    pts_nf = p0.copy()
    alive_nf = np.ones(n_init, dtype=bool)

    pts_fb = p0.copy()
    alive_fb = np.ones(n_init, dtype=bool)
    trajectories = [deque([tuple(p0[i, 0])], maxlen=20) for i in range(n_init)]

    writer, codec_used = make_video_writer(out_video_path, fps, (w, h))

    frame_records = []
    vis0 = frames[0].copy()
    for i in range(n_init):
        x, y = int(round(pts_fb[i, 0, 0])), int(round(pts_fb[i, 0, 1]))
        col = (0, 220, 0) if is_obj[i] else (220, 180, 0)
        cv2.circle(vis0, (x, y), 6, col, -1)
    writer.write(vis0)

    frame_records.append([
        video_tag, 0,
        n_init, n_init,
        n_init_obj, n_init_obj,
        n_init_bg, n_init_bg,
        0, 0.0
    ])

    t_fwd_total = 0.0
    t_fb_total = 0.0
    g_prev = g0

    snap_vis = {}
    fb_errors_dropped = []
    fb_errors_kept = []

    for f_idx in range(1, len(frames)):
        g_cur = cv2.cvtColor(frames[f_idx], cv2.COLOR_BGR2GRAY)

        # 1) Без зворотної фільтрації
        t0 = time.perf_counter()
        idx_nf = np.where(alive_nf)[0]
        if len(idx_nf) > 0:
            p_next, st, _ = cv2.calcOpticalFlowPyrLK(g_prev, g_cur, pts_nf[idx_nf], None, **lk_params)
            t_fwd_total += (time.perf_counter() - t0)
            st_ok = (st.ravel() == 1)
            xy = p_next.reshape(-1, 2)
            in_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            valid = st_ok & in_bounds
            pts_nf[idx_nf[valid]] = p_next[valid]
            alive_nf[idx_nf[~valid]] = False
        else:
            t_fwd_total += (time.perf_counter() - t0)

        # 2) Зі зворотною перевіркою (поріг fb_err <= 1.0 пк)
        t1 = time.perf_counter()
        idx_fb = np.where(alive_fb)[0]
        dropped_this = 0
        mean_fb_this = 0.0
        just_dropped_xy = []
        if len(idx_fb) > 0:
            p_next, st1, _ = cv2.calcOpticalFlowPyrLK(g_prev, g_cur, pts_fb[idx_fb], None, **lk_params)
            p_back, st2, _ = cv2.calcOpticalFlowPyrLK(g_cur, g_prev, p_next, None, **lk_params)
            t_fb_total += (time.perf_counter() - t1)

            fb_err = np.linalg.norm((pts_fb[idx_fb] - p_back).reshape(-1, 2), axis=1)
            xy = p_next.reshape(-1, 2)
            in_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            ok = (st1.ravel() == 1) & (st2.ravel() == 1) & in_bounds & (fb_err <= 1.0)

            mean_fb_this = float(fb_err[ok].mean()) if ok.any() else 0.0
            if ok.any():
                fb_errors_kept.extend(fb_err[ok].tolist())
            if (~ok).any():
                fb_errors_dropped.extend(fb_err[~ok].tolist())
                dropped_this = int((~ok).sum())
                just_dropped_xy = xy[~ok]

            for k_local, g_i in enumerate(idx_fb):
                if ok[k_local]:
                    pts_fb[g_i] = p_next[k_local]
                    trajectories[g_i].append((float(xy[k_local, 0]), float(xy[k_local, 1])))
                else:
                    alive_fb[g_i] = False
        else:
            t_fb_total += (time.perf_counter() - t1)

        # Малювання траєкторій (останні 20 позицій) поверх кадру
        vis = frames[f_idx].copy()
        for g_i in np.where(alive_fb)[0]:
            tr = list(trajectories[g_i])
            is_fg_pt = is_obj[g_i]
            line_col = (0, 235, 110) if is_fg_pt else (220, 175, 40)
            pt_col = (0, 255, 255) if is_fg_pt else (255, 200, 60)
            thick = 4 if is_fg_pt else 2
            for s in range(1, len(tr)):
                p_a = (int(round(tr[s - 1][0])), int(round(tr[s - 1][1])))
                p_b = (int(round(tr[s][0])), int(round(tr[s][1])))
                cv2.line(vis, p_a, p_b, line_col, thick, cv2.LINE_AA)
            cv2.circle(vis, (int(round(tr[-1][0])), int(round(tr[-1][1]))), 7 if is_fg_pt else 4, pt_col, -1)

        for dx, dy in just_dropped_xy:
            if 0 <= dx < w and 0 <= dy < h:
                cv2.drawMarker(vis, (int(round(dx)), int(round(dy))), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 3)

        if f_idx in snap_frames:
            snap_vis[f_idx] = vis.copy()

        cv2.rectangle(vis, (24, 24), (1050, 92), (20, 20, 20), -1)
        cv2.putText(
            vis,
            f"Frame {f_idx:3d} | No-filter: {int(alive_nf.sum())}/{n_init} | FB<=1px: {int(alive_fb.sum())}/{n_init}",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA
        )
        writer.write(vis)

        frame_records.append([
            video_tag, f_idx,
            int(alive_nf.sum()), int(alive_fb.sum()),
            int((alive_nf & is_obj).sum()), int((alive_fb & is_obj).sum()),
            int((alive_nf & ~is_obj).sum()), int((alive_fb & ~is_obj).sum()),
            dropped_this, f"{mean_fb_this:.4f}"
        ])
        g_prev = g_cur

    writer.release()

    n_pairs = len(frames) - 1
    ms_fwd = (t_fwd_total * 1000.0) / n_pairs
    ms_fb = (t_fb_total * 1000.0) / n_pairs

    nf_end = int(alive_nf.sum())
    fb_end = int(alive_fb.sum())
    nf_obj_end = int((alive_nf & is_obj).sum())
    fb_obj_end = int((alive_fb & is_obj).sum())
    nf_bg_end = int((alive_nf & ~is_obj).sum())
    fb_bg_end = int((alive_fb & ~is_obj).sum())

    summary_rows = [
        [
            video_tag, "Без фільтрації (лише прямий PyrLK)", n_init,
            nf_end, f"{nf_end / n_init * 100:.1f}",
            n_init_obj, nf_obj_end, f"{nf_obj_end / max(n_init_obj, 1) * 100:.1f}",
            n_init_bg, nf_bg_end, f"{nf_bg_end / max(n_init_bg, 1) * 100:.1f}",
            f"{ms_fwd:.2f}", codec_used
        ],
        [
            video_tag, "Із зворотною перевіркою (e_FB <= 1.0 пк)", n_init,
            fb_end, f"{fb_end / n_init * 100:.1f}",
            n_init_obj, fb_obj_end, f"{fb_obj_end / max(n_init_obj, 1) * 100:.1f}",
            n_init_bg, fb_bg_end, f"{fb_bg_end / max(n_init_bg, 1) * 100:.1f}",
            f"{ms_fb:.2f}", codec_used
        ],
    ]

    info = {
        "n_init": n_init,
        "n_init_obj": n_init_obj,
        "n_init_bg": n_init_bg,
        "ms_fwd": ms_fwd,
        "ms_fb": ms_fb,
        "records": frame_records,
        "summary_rows": summary_rows,
        "snap_vis": snap_vis,
        "fb_kept_mean": float(np.mean(fb_errors_kept)) if fb_errors_kept else 0.0,
        "fb_dropped_median": float(np.median(fb_errors_dropped)) if fb_errors_dropped else 0.0,
    }
    return info


def run_part_a_b():
    print("=" * 65)
    print("Частини A і B: Аналіз відео та розріджений оптичний потік (Кроки 1–7)")
    print("=" * 65)

    # Послідовне завантаження (одне відео за раз задля економії ОЗП при 2592x1944)
    frames_a, fps_a, wa, ha = load_video_frames(VIDEO_A_PATH)
    bg_a = build_clean_background(frames_a, mode="a")
    row_a, ma_a = analyze_exposure(frames_a, fps_a, "Відео A", "one_object.mp4")
    rep_a = frames_a[40].copy()
    print(f"Відео A ({wa}x{ha}): {len(frames_a)} кадрів ({len(frames_a)/fps_a:.2f} с, {fps_a:.2f} fps), "
          f"яскравість = {ma_a.mean():.2f} +- {ma_a.std():.2f} (розмах {ma_a.max()-ma_a.min():.2f})")
    info_a = run_sparse_lk(frames_a, bg_a, "Відео A", RESULTS_DIR / "b_trajectories_video_a.mp4", fps_a, snap_frames=(160,))
    del frames_a, bg_a
    gc.collect()

    frames_b, fps_b, wb, hb = load_video_frames(VIDEO_B_PATH)
    bg_b = build_clean_background(frames_b, mode="b")
    row_b, ma_b = analyze_exposure(frames_b, fps_b, "Відео B", "two_objects.mp4")
    rep_b = frames_b[120].copy()
    print(f"Відео B ({wb}x{hb}): {len(frames_b)} кадрів ({len(frames_b)/fps_b:.2f} с, {fps_b:.2f} fps), "
          f"яскравість = {ma_b.mean():.2f} +- {ma_b.std():.2f} (розмах {ma_b.max()-ma_b.min():.2f})")
    info_b = run_sparse_lk(frames_b, bg_b, "Відео B", RESULTS_DIR / "b_trajectories_video_b.mp4", fps_b, snap_frames=(200,))
    del frames_b, bg_b
    gc.collect()

    save_csv(
        RESULTS_DIR / "a_video_stats.csv",
        [
            "video_id", "file_name", "resolution", "fps", "frames",
            "duration_s", "mean_gray", "std_gray", "min_gray", "max_gray", "range_gray"
        ],
        [row_a, row_b]
    )

    # Рисунок a_exposure_stability.png
    fig = plt.figure(figsize=(11.2, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(cv2.cvtColor(rep_a, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"(а) Відео A ({wa}×{ha}, кадр № 40): рух одного об'єкта", fontsize=9.8)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(cv2.cvtColor(rep_b, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"(б) Відео B ({wb}×{hb}, кадр № 120): два об'єкти з перекриттям", fontsize=9.8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 0])
    ta = np.arange(len(ma_a)) / fps_a
    ax2.plot(ta, ma_a, color="#1b9e77", lw=1.6, label=f"Середня яскравість кадру ({wa}×{ha})")
    ax2.axhline(ma_a.mean(), color="#333333", ls="--", lw=1.1, label=f"Середнє ({ma_a.mean():.2f})")
    ax2.set_ylim(118, 134)
    ax2.set_xlabel("Час запису $t$, с")
    ax2.set_ylabel("Середня яскравість, рівні [0..255]")
    ax2.set_title(f"(в) Стабільність експозиції Відео A ($\\sigma = {ma_a.std():.2f}$, $\\Delta = {ma_a.max()-ma_a.min():.2f}$)", fontsize=9.5)
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8.5)

    ax3 = fig.add_subplot(gs[1, 1])
    tb = np.arange(len(ma_b)) / fps_b
    ax3.plot(tb, ma_b, color="#d95f02", lw=1.6, label=f"Середня яскравість кадру ({wb}×{hb})")
    ax3.axhline(ma_b.mean(), color="#333333", ls="--", lw=1.1, label=f"Середнє ({ma_b.mean():.2f})")
    ax3.set_ylim(118, 134)
    ax3.set_xlabel("Час запису $t$, с")
    ax3.set_ylabel("Середня яскравість, рівні [0..255]")
    ax3.set_title(f"(г) Стабільність експозиції Відео B ($\\sigma = {ma_b.std():.2f}$, $\\Delta = {ma_b.max()-ma_b.min():.2f}$)", fontsize=9.5)
    ax3.grid(True, ls=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "a_exposure_stability.png")
    plt.close(fig)

    print("\nРезультати розрідженого оптичного потоку PyrLK:")
    for tag, inf in [("Відео A", info_a), ("Відео B", info_b)]:
        for r in inf["summary_rows"]:
            print(f"  {tag} | {r[1]}: всього {r[3]}/{r[2]} ({r[4]}%), об'єкт {r[6]}/{r[5]} ({r[7]}%), фон {r[9]}/{r[8]} ({r[10]}%), {r[11]} мс/кадр")

    save_csv(
        RESULTS_DIR / "b_lk_summary.csv",
        [
            "video_id", "mode", "init_total", "final_total", "surv_total_pct",
            "init_obj", "final_obj", "surv_obj_pct",
            "init_bg", "final_bg", "surv_bg_pct", "time_ms_per_frame", "codec"
        ],
        info_a["summary_rows"] + info_b["summary_rows"]
    )

    save_csv(
        RESULTS_DIR / "b_lk_survival_frames.csv",
        ["video_id", "frame_idx", "alive_nofilt", "alive_fb", "obj_nofilt", "obj_fb", "bg_nofilt", "bg_fb", "dropped_fb", "mean_fb_err_px"],
        info_a["records"] + info_b["records"]
    )

    rec_a = np.array([[int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5])] for r in info_a["records"]])
    rec_b = np.array([[int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5])] for r in info_b["records"]])

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    l1, = axl.plot(rec_a[:, 0], rec_a[:, 1], color="#d95f02", lw=2.0, label="Усі точки без фільтрації")
    l2, = axl.plot(rec_a[:, 0], rec_a[:, 2], color="#1b9e77", lw=2.0, label="Усі точки + FB ($e_{\\mathrm{FB}} \\leq 1$ пк)")
    l3, = axl.plot(rec_a[:, 0], rec_a[:, 3], color="#d95f02", lw=1.4, ls="--", label="На об'єкті (без фільтрації)")
    l4, = axl.plot(rec_a[:, 0], rec_a[:, 4], color="#1b9e77", lw=1.4, ls="--", label="На об'єкті + FB ($e_{\\mathrm{FB}} \\leq 1$ пк)")
    axl.set_xlabel("Номер кадру")
    axl.set_ylabel("Кількість живих точок, од.")
    axl.set_title(f"(а) Виживання точок LK: Відео A ($N_0 = {info_a['n_init']}$)")
    axl.grid(True, ls=":", alpha=0.6)

    axr.plot(rec_b[:, 0], rec_b[:, 1], color="#d95f02", lw=2.0)
    axr.plot(rec_b[:, 0], rec_b[:, 2], color="#1b9e77", lw=2.0)
    axr.plot(rec_b[:, 0], rec_b[:, 3], color="#d95f02", lw=1.4, ls="--")
    axr.plot(rec_b[:, 0], rec_b[:, 4], color="#1b9e77", lw=1.4, ls="--")
    axr.set_xlabel("Номер кадру")
    axr.set_ylabel("Кількість живих точок, од.")
    axr.set_title(f"(б) Виживання точок LK: Відео B ($N_0 = {info_b['n_init']}$)")
    axr.grid(True, ls=":", alpha=0.6)

    fig.legend([l1, l2, l3, l4],
               ["Усі точки без фільтрації", "Усі точки + FB ($e_{\\mathrm{FB}} \\leq 1$ пк)",
                "На рухомому об'єкті (без фільтрації)", "На рухомому об'єкті + FB ($e_{\\mathrm{FB}} \\leq 1$ пк)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(FIGURES_DIR / "b_lk_survival.png")
    plt.close(fig)

    snap_a = info_a["snap_vis"][160]
    snap_b = info_b["snap_vis"][200]
    fig, (ax_va, ax_vb) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax_va.imshow(cv2.cvtColor(snap_a, cv2.COLOR_BGR2RGB))
    ax_va.set_title("(а) Відео A, кадр № 160: 20-кадрові траєкторії PyrLK", fontsize=10)
    ax_va.axis("off")

    ax_vb.imshow(cv2.cvtColor(snap_b, cv2.COLOR_BGR2RGB))
    ax_vb.set_title("(б) Відео B, кадр № 200: траєкторії двох об'єктів", fontsize=10)
    ax_vb.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "b_lk_trajectories.png")
    plt.close(fig)

    print("Частини A і B успішно виконано!")


if __name__ == "__main__":
    run_part_a_b()
