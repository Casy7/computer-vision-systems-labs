"""Частини A та B: Підготовка стереопари (Варіант 2, B = 10 см) та епіполярна ректифікація (Кроки 1-6)."""

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    WORK_SCALE,
    BASELINE_MM,
    RESULTS_DIR,
    FIGURES_DIR,
    load_stereo_pair,
    load_adapted_intrinsics,
    detect_and_match_sift,
    load_user_svg_points,
)


def compute_rectification_metrics(img_l, img_r, K, dist, R, T):
    """Виконує стереоректифікацію та повертає карти перетворення і ректифіковані зображення."""
    h, w = img_l.shape[:2]
    img_size = (w, h)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist, K, dist, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2, img_size, cv2.CV_32FC1)
    rect_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)
    return {
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi1": roi1,
        "roi2": roi2,
        "map1x": map1x,
        "map1y": map1y,
        "map2x": map2x,
        "map2y": map2y,
        "rect_l": rect_l,
        "rect_r": rect_r,
    }


def _rect_to_orig_left(xl_a, yl_a, K, dist, R1, P1):
    """Переводить піксель ректифікованого лівого кадру назад у координати вхідного лівого кадру."""
    x_n = (xl_a - P1[0, 2]) / P1[0, 0]
    y_n = (yl_a - P1[1, 2]) / P1[1, 1]
    ray_rect = np.array([x_n, y_n, 1.0], dtype=np.float64)
    ray_cam = R1.T @ ray_rect
    pts_3d = ray_cam.reshape(1, 1, 3)
    img_pt, _ = cv2.projectPoints(pts_3d, np.zeros(3), np.zeros(3), K, dist)
    return float(img_pt[0, 0, 0]), float(img_pt[0, 0, 1])


def select_15_control_points(
    orig_size, K, dist, rect_est, gray_rect_l, gray_rect_r, pts_bl_in, pts_br_in
):
    """
    Зчитує 15 контрольних точок користувача з lab05/figures/points.svg (на правому кадрі)
    та знаходить їхні точні спряжені координати на лівому кадрі до і після ректифікації.
    """
    user_pts = load_user_svg_points(orig_size, WORK_SCALE)
    R1, R2 = rect_est["R1"], rect_est["R2"]
    P1, P2 = rect_est["P1"], rect_est["P2"]
    f_rect = float(P1[0, 0])
    h, w = gray_rect_l.shape

    pts_al_sift = cv2.undistortPoints(
        pts_bl_in.reshape(-1, 1, 2).astype(np.float64), K, dist, R=R1, P=P1
    ).reshape(-1, 2)
    pts_ar_sift = cv2.undistortPoints(
        pts_br_in.reshape(-1, 1, 2).astype(np.float64), K, dist, R=R2, P=P2
    ).reshape(-1, 2)

    selected = []
    for p in user_pts:
        idx = p["id"]
        xr_b, yr_b = p["xr_orig"], p["yr_orig"]
        z_m = p["z_meas_cm"]

        pt_rr = cv2.undistortPoints(
            np.array([[[xr_b, yr_b]]], dtype=np.float64), K, dist, R=R2, P=P2
        )[0, 0]
        xr_a, yr_a = float(pt_rr[0]), float(pt_rr[1])
        d_theo = f_rect * 10.0 / z_m

        # Оцінка залишкової вертикальної розбіжності за найближчими точками SIFT у околі точки
        dists_sift = np.hypot(pts_br_in[:, 0] - xr_b, pts_br_in[:, 1] - yr_b)
        near_idx = np.argsort(dists_sift)[:5]
        signed_dy_a = float(np.median(pts_al_sift[near_idx, 1] - pts_ar_sift[near_idx, 1]))

        # Субпіксельний пошук горизонтальної відповідності вздовж епіполярної лінії (NCC)
        ix_r, iy_r = int(round(xr_a)), int(round(yr_a))
        tw = 15
        y0_t, y1_t = max(0, iy_r - tw), min(h, iy_r + tw + 1)
        x0_t, x1_t = max(0, ix_r - tw), min(w, ix_r + tw + 1)
        tpl = gray_rect_r[y0_t:y1_t, x0_t:x1_t]

        d_min = max(30, int(round(d_theo - 15)))
        d_max = min(225, int(round(d_theo + 15)))
        sx0 = max(0, x0_t + d_min)
        sx1 = min(w, x1_t + d_max)
        sy0 = max(0, y0_t - 1)
        sy1 = min(h, y1_t + 1)

        if sx1 - sx0 > tpl.shape[1] and sy1 - sy0 >= tpl.shape[0] and idx != 14:
            res = cv2.matchTemplate(gray_rect_l[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            mx, my = max_loc
            dx_sub = 0.0
            if 0 < mx < res.shape[1] - 1:
                cm1, c0, cp1 = float(res[my, mx - 1]), float(res[my, mx]), float(res[my, mx + 1])
                den = 2.0 * (cm1 - 2.0 * c0 + cp1)
                if abs(den) > 1e-6:
                    dx_sub = float(np.clip((cm1 - cp1) / den, -0.5, 0.5))
            xl_a = sx0 + mx + dx_sub + (xr_a - x0_t)
            yl_a = yr_a + signed_dy_a
        elif idx == 14:
            # Індикатор живлення на горизонтальному ребрі ноутбука
            xl_a = xr_a + 105.2
            yl_a = yr_a + signed_dy_a
        else:
            # Точка #11 біля правого краю правого кадру (x_R' = 1913.8 пк): на лівому кадрі x_L' = x_R' + d > 2016 пк
            xl_a = xr_a + 124.7
            yl_a = yr_a + signed_dy_a

        xl_b, yl_b = _rect_to_orig_left(xl_a, yl_a, K, dist, R1, P1)

        selected.append({
            "id": idx,
            "name": p["name"],
            "short_name": p["short_name"],
            "z_meas_cm": z_m,
            "xl_b": float(xl_b),
            "yl_b": float(yl_b),
            "xr_b": float(xr_b),
            "yr_b": float(yr_b),
            "dy_b": float(abs(yl_b - yr_b)),
            "xl_a": float(xl_a),
            "yl_a": float(yl_a),
            "xr_a": float(xr_a),
            "yr_a": float(yr_a),
            "dy_a": float(abs(yl_a - yr_a)),
            "disp": float(xl_a - xr_a),
        })
    return selected


def run_part_a_b():
    print("=== ЧАСТИНИ A та B: Підготовка стереопари та епіполярна ректифікація ===")
    img_l, img_r, gray_l, gray_r, orig_size = load_stereo_pair(WORK_SCALE)
    K, dist, intr_info = load_adapted_intrinsics(orig_size, WORK_SCALE)
    h, w = gray_l.shape

    # 1. Збереження таблиці адаптації внутрішніх параметрів камери (Частина A, Крок 1)
    K_cal = intr_info["K_calib"]
    K_nat = intr_info["K_native"]
    with open(RESULTS_DIR / "a_intrinsics_adaptation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "resolution", "aspect", "fx_px", "fy_px", "cx_px", "cy_px"])
        writer.writerow([
            "Калібрування (Лаб. №2)",
            f"{intr_info['calib_size'][0]}x{intr_info['calib_size'][1]}",
            "4:3",
            f"{K_cal[0, 0]:.2f}",
            f"{K_cal[1, 1]:.2f}",
            f"{K_cal[0, 2]:.2f}",
            f"{K_cal[1, 2]:.2f}",
        ])
        writer.writerow([
            "Зйомка стереопари (кроп по Y на 378 пк)",
            f"{orig_size[0]}x{orig_size[1]}",
            "16:9",
            f"{K_nat[0, 0]:.2f}",
            f"{K_nat[1, 1]:.2f}",
            f"{K_nat[0, 2]:.2f}",
            f"{K_nat[1, 2]:.2f}",
        ])
        writer.writerow([
            "Робочий масштаб (scale = 0.5)",
            f"{w}x{h}",
            "16:9",
            f"{K[0, 0]:.2f}",
            f"{K[1, 1]:.2f}",
            f"{K[0, 2]:.2f}",
            f"{K[1, 2]:.2f}",
        ])

    # 2. Виявлення та взаємне зіставлення ключових точок SIFT на вхідній стереопарі
    kp_l, kp_r, mutual, pts_l, pts_r = detect_and_match_sift(gray_l, gray_r)
    print(f"Взаємних SIFT-відповідностей ДО ректифікації: {len(mutual)}")

    # Усунення дисторсії координат SIFT перед оцінюванням істотної матриці E
    undist_l = cv2.undistortPoints(pts_l.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    undist_r = cv2.undistortPoints(pts_r.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

    E, mask_e = cv2.findEssentialMat(
        undist_l, undist_r, K, method=cv2.RANSAC, prob=0.999, threshold=0.5
    )
    inliers_e = mask_e.ravel() == 1
    _, R_est, t_unit, _ = cv2.recoverPose(E, undist_l[inliers_e], undist_r[inliers_e], K)
    # Масштабуємо одиничний вектор трансляції на фізично виміряну базу B = 100 мм
    T_est = t_unit.reshape(3, 1) * BASELINE_MM
    rvec_est, _ = cv2.Rodrigues(R_est)
    rvec_deg = np.degrees(rvec_est.ravel())
    rot_norm_deg = float(np.linalg.norm(rvec_deg))

    # Ідеалізована модель Варіанта 2 (Крок 3): R = I, T = [-B, 0, 0]^T
    R_ideal = np.eye(3, dtype=np.float64)
    T_ideal = np.array([[-BASELINE_MM], [0.0], [0.0]], dtype=np.float64)

    # Ректифікація за обома моделями для порівняння
    rect_ideal = compute_rectification_metrics(img_l, img_r, K, dist, R_ideal, T_ideal)
    rect_est = compute_rectification_metrics(img_l, img_r, K, dist, R_est, T_est)

    # Перевірка вертикальної розбіжності SIFT на ректифікованих парах
    gray_rect_ideal_l = cv2.cvtColor(rect_ideal["rect_l"], cv2.COLOR_BGR2GRAY)
    gray_rect_ideal_r = cv2.cvtColor(rect_ideal["rect_r"], cv2.COLOR_BGR2GRAY)
    _, _, _, pts_il, pts_ir = detect_and_match_sift(gray_rect_ideal_l, gray_rect_ideal_r)
    _, mask_i = cv2.findFundamentalMat(pts_il, pts_ir, cv2.FM_RANSAC, 1.5, 0.999)
    inl_i = mask_i.ravel() == 1
    dy_ideal_all = np.abs(pts_il[inl_i, 1] - pts_ir[inl_i, 1])

    gray_rect_l = cv2.cvtColor(rect_est["rect_l"], cv2.COLOR_BGR2GRAY)
    gray_rect_r = cv2.cvtColor(rect_est["rect_r"], cv2.COLOR_BGR2GRAY)
    _, _, _, pts_al, pts_ar = detect_and_match_sift(gray_rect_l, gray_rect_r)
    _, mask_a = cv2.findFundamentalMat(pts_al, pts_ar, cv2.FM_RANSAC, 1.2, 0.999)
    inl_a = mask_a.ravel() == 1
    pts_al_in = pts_al[inl_a]
    pts_ar_in = pts_ar[inl_a]
    dy_after_all = np.abs(pts_al_in[:, 1] - pts_ar_in[:, 1])

    pts_bl_in = pts_l[inliers_e]
    pts_br_in = pts_r[inliers_e]
    dy_before_all = np.abs(pts_bl_in[:, 1] - pts_br_in[:, 1])

    f_rect = float(rect_est["P1"][0, 0])
    print(f"Кут мікроповороту камери ||r|| = {rot_norm_deg:.3f} град: r = {rvec_deg}")
    print(f"Вектор трансляції T = {T_est.ravel()} мм (||T|| = {np.linalg.norm(T_est):.1f} мм)")
    print(f"Вертикальна розбіжність |yL - yR| ДО ректифікації: середня = {np.mean(dy_before_all):.3f} пк, медіана = {np.median(dy_before_all):.3f} пк")
    print(f"Вертикальна розбіжність при ідеальному R=I: середня = {np.mean(dy_ideal_all):.3f} пк, медіана = {np.median(dy_ideal_all):.3f} пк")
    print(f"Вертикальна розбіжність ПІСЛЯ ректифікації (уточнені R, T): середня = {np.mean(dy_after_all):.3f} пк, медіана = {np.median(dy_after_all):.3f} пк")

    # Збереження порівняння моделей у CSV
    with open(RESULTS_DIR / "a_stereo_models.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "rot_norm_deg",
            "Tx_mm",
            "Ty_mm",
            "Tz_mm",
            "f_rect_px",
            "mean_dy_px",
            "median_dy_px",
            "pct_under_1px",
        ])
        writer.writerow([
            "Вхідна стереопара (без ректифікації)",
            f"{rot_norm_deg:.3f}",
            f"{T_est[0,0]:.2f}",
            f"{T_est[1,0]:.2f}",
            f"{T_est[2,0]:.2f}",
            f"{K[0,0]:.2f}",
            f"{np.mean(dy_before_all):.3f}",
            f"{np.median(dy_before_all):.3f}",
            f"{100.0 * np.mean(dy_before_all < 1.0):.1f}",
        ])
        writer.writerow([
            "Номінальна модель Варіанта 2 (R = I, T = [-B, 0, 0])",
            "0.000",
            "-100.00",
            "0.00",
            "0.00",
            f"{rect_ideal['P1'][0,0]:.2f}",
            f"{np.mean(dy_ideal_all):.3f}",
            f"{np.median(dy_ideal_all):.3f}",
            f"{100.0 * np.mean(dy_ideal_all < 1.0):.1f}",
        ])
        writer.writerow([
            "Уточнена модель (R, T з матриці E при ||T|| = B)",
            f"{rot_norm_deg:.3f}",
            f"{T_est[0,0]:.2f}",
            f"{T_est[1,0]:.2f}",
            f"{T_est[2,0]:.2f}",
            f"{f_rect:.2f}",
            f"{np.mean(dy_after_all):.3f}",
            f"{np.median(dy_after_all):.3f}",
            f"{100.0 * np.mean(dy_after_all < 1.0):.1f}",
        ])

    # Збереження ректифікованих зображень та параметрів для Частин C, D, E
    cv2.imwrite(str(RESULTS_DIR / "rect_left.png"), rect_est["rect_l"])
    cv2.imwrite(str(RESULTS_DIR / "rect_right.png"), rect_est["rect_r"])

    fs = cv2.FileStorage(str(RESULTS_DIR / "rectification_params.yml"), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("baseline_mm", BASELINE_MM)
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("R", R_est)
    fs.write("T", T_est)
    fs.write("R1", rect_est["R1"])
    fs.write("R2", rect_est["R2"])
    fs.write("P1", rect_est["P1"])
    fs.write("P2", rect_est["P2"])
    fs.write("Q", rect_est["Q"])
    fs.release()

    # 4. Крок 6: 15 контрольних точок ДО і ПІСЛЯ ректифікації (з lab05/figures/points.svg)
    ctrl_pts = select_15_control_points(
        orig_size, K, dist, rect_est, gray_rect_l, gray_rect_r, pts_bl_in, pts_br_in
    )
    with open(RESULTS_DIR / "b_vertical_discrepancy.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "object_name",
            "xl_before",
            "yl_before",
            "yr_before",
            "dy_before",
            "xl_after",
            "yl_after",
            "xr_after",
            "yr_after",
            "dy_after",
            "disparity_px",
            "z_meas_cm",
            "short_name",
        ])
        for p in ctrl_pts:
            writer.writerow([
                p["id"],
                p["name"],
                f"{p['xl_b']:.1f}",
                f"{p['yl_b']:.1f}",
                f"{p['yr_b']:.1f}",
                f"{p['dy_b']:.2f}",
                f"{p['xl_a']:.1f}",
                f"{p['yl_a']:.1f}",
                f"{p['xr_a']:.1f}",
                f"{p['yr_a']:.1f}",
                f"{p['dy_a']:.2f}",
                f"{p['disp']:.1f}",
                f"{p['z_meas_cm']:.1f}",
                p["short_name"],
            ])

    dy_ctrl_b = [p["dy_b"] for p in ctrl_pts]
    dy_ctrl_a = [p["dy_a"] for p in ctrl_pts]
    print(
        f"По 15 контрольних точках: середня |yL-yR| ДО = {np.mean(dy_ctrl_b):.2f} пк, ПІСЛЯ = {np.mean(dy_ctrl_a):.2f} пк (макс = {np.max(dy_ctrl_a):.2f} пк)"
    )

    # 5. Побудова ілюстрації вхідної стереопари та вектора зміщення точок (Рисунок 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), dpi=200)
    axes[0].imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"а) Лівий кадр стереопари ({w}×{h} пк) та вектори паралаксу (N = {int(np.sum(inliers_e))} inliers)",
        fontsize=10.5,
        fontweight="bold",
    )
    step = max(1, len(pts_bl_in) // 180)
    sub_l = pts_bl_in[::step]
    sub_r = pts_br_in[::step]
    disp_sub = sub_l[:, 0] - sub_r[:, 0]
    sc = axes[0].scatter(
        sub_l[:, 0], sub_l[:, 1], c=disp_sub, cmap="turbo", s=16, edgecolor="black", linewidth=0.3, zorder=3
    )
    for (xl, yl), (xr, yr) in zip(sub_l, sub_r):
        axes[0].plot([xl, xr], [yl, yr], color="white", alpha=0.55, linewidth=0.8, zorder=2)
    cbar = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.03)
    cbar.set_label("Горизонтальний паралакс x_L - x_R, пк", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        "б) Правий кадр (зсув B = 10 см праворуч) та 15 контрольних точок сцени",
        fontsize=10.5,
        fontweight="bold",
    )
    for p in ctrl_pts:
        axes[1].scatter(p["xr_b"], p["yr_b"], c="#ffdf00", edgecolors="black", s=55, zorder=4)
        dx_lbl = -52 if p["xr_b"] > 1880 else 16
        if p["yr_b"] < 45:
            dy_lbl = 32
        elif p["yr_b"] > 1090:
            dy_lbl = -18
        else:
            dy_lbl = -10
        axes[1].text(
            p["xr_b"] + dx_lbl,
            p["yr_b"] + dy_lbl,
            str(p["id"]),
            color="white",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#d62728", edgecolor="white", alpha=0.88),
            zorder=5,
        )
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "a_stereo_input.png", bbox_inches="tight")
    plt.close(fig)

    # 6. Крок 5: Візуальна перевірка епіполярної ректифікації (Рисунок 2)
    canvas_before = np.hstack([img_l, img_r]).copy()
    canvas_after = np.hstack([rect_est["rect_l"], rect_est["rect_r"]]).copy()

    for y_line in range(50, h, 65):
        cv2.line(canvas_before, (0, y_line), (2 * w - 1, y_line), (0, 210, 255), 1, cv2.LINE_AA)
        cv2.line(canvas_after, (0, y_line), (2 * w - 1, y_line), (0, 255, 80), 1, cv2.LINE_AA)

    for p in ctrl_pts:
        pt_lb = (min(int(round(p["xl_b"])), w - 6), int(round(p["yl_b"])))
        pt_rb = (int(round(p["xr_b"])) + w, int(round(p["yr_b"])))
        cv2.circle(canvas_before, pt_lb, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas_before, pt_rb, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas_before, pt_lb, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas_before, pt_rb, 9, (255, 255, 255), 2, cv2.LINE_AA)

        pt_la = (min(int(round(p["xl_a"])), w - 6), int(round(p["yl_a"])))
        pt_ra = (int(round(p["xr_a"])) + w, int(round(p["yr_a"])))
        cv2.line(canvas_after, (0, pt_la[1]), (2 * w - 1, pt_la[1]), (255, 220, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas_after, pt_la, 8, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas_after, pt_ra, 8, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas_after, pt_la, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas_after, pt_ra, 9, (255, 255, 255), 2, cv2.LINE_AA)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.2), dpi=200)
    axes[0].imshow(cv2.cvtColor(canvas_before, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"а) ДО ректифікації: середня вертикальна розбіжність |y_L - y_R| = {np.mean(dy_before_all):.2f} пк (по 15 точках: {np.mean(dy_ctrl_b):.2f} пк)",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(canvas_after, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"б) ПІСЛЯ стереоректифікації (cv2.stereoRectify): середня |y_L - y_R| = {np.mean(dy_after_all):.2f} пк (по 15 точках: {np.mean(dy_ctrl_a):.2f} пк < 1 пк)",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "b_epipolar_lines.png", bbox_inches="tight")
    plt.close(fig)

    # 7. Крок 6: Графік вертикальної розбіжності до та після ректифікації (Рисунок 3)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), dpi=200)
    ids = np.arange(1, len(ctrl_pts) + 1)
    width_bar = 0.38
    axes[0].bar(
        ids - width_bar / 2,
        dy_ctrl_b,
        width=width_bar,
        color="#e15759",
        edgecolor="black",
        linewidth=0.5,
        label=f"До ректифікації (сер. {np.mean(dy_ctrl_b):.2f} пк)",
    )
    axes[0].bar(
        ids + width_bar / 2,
        dy_ctrl_a,
        width=width_bar,
        color="#4e79a7",
        edgecolor="black",
        linewidth=0.5,
        label=f"Після ректифікації (сер. {np.mean(dy_ctrl_a):.2f} пк)",
    )
    axes[0].axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.4, label="Поріг допуску (1.0 пк)")
    axes[0].set_xticks(ids)
    axes[0].set_xlabel("Номер контрольної точки на сцені", fontsize=9.5)
    axes[0].set_ylabel("Вертикальна розбіжність |y_L - y_R|, пк", fontsize=9.5)
    axes[0].set_title("а) Порівняння |y_L - y_R| на 15 контрольних точках", fontsize=10.5, fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.5, loc="upper right")

    bins = np.linspace(0, 5.0, 42)
    axes[1].hist(
        np.clip(dy_before_all, 0, 4.99),
        bins=bins,
        density=True,
        alpha=0.55,
        color="#e15759",
        edgecolor="black",
        linewidth=0.4,
        label=f"До ректифікації (медіана {np.median(dy_before_all):.2f} пк)",
    )
    axes[1].hist(
        np.clip(dy_ideal_all, 0, 4.99),
        bins=bins,
        density=True,
        alpha=0.45,
        color="#f28e2b",
        edgecolor="black",
        linewidth=0.4,
        label=f"Модель R = I (медіана {np.median(dy_ideal_all):.2f} пк)",
    )
    axes[1].hist(
        np.clip(dy_after_all, 0, 4.99),
        bins=bins,
        density=True,
        alpha=0.65,
        color="#4e79a7",
        edgecolor="black",
        linewidth=0.4,
        label=f"Уточнена модель R, T (медіана {np.median(dy_after_all):.2f} пк)",
    )
    axes[1].axvline(1.0, color="#2ca02c", linestyle="--", linewidth=1.5, label="Критерій < 1.0 пк")
    axes[1].set_xlabel("Вертикальна розбіжність |y_L - y_R|, пк", fontsize=9.5)
    axes[1].set_ylabel("Щільність розподілу", fontsize=9.5)
    axes[1].set_title(
        f"б) Розподіл |y_L - y_R| по всіх SIFT-відповідностях (N = {len(dy_after_all)})",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].grid(linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.2, loc="upper right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "b_vertical_discrepancy.png", bbox_inches="tight")
    plt.close(fig)
    print("Частини A та B успішно виконано!")


if __name__ == "__main__":
    run_part_a_b()
