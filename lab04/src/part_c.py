"""
Лабораторна робота № 4. Частина C: Власний алгоритм RANSAC (Кроки 8-12).
- Крок 8: Реалізація схеми RANSAC (вибірка 4 пар, H через getPerspectiveTransform, підрахунок inliers, МНК-уточнення);
- Крок 9: Кількість ітерацій N: теоретичне N для w = 0.5 та для спостережуваної частки викидів,
          фіксоване N = 1000, адаптивна зупинка; статистика за n_runs незалежних запусків;
- Крок 10: Порівняння власної реалізації з cv2.findHomography(..., cv2.RANSAC, tau);
- Крок 11: Дослідження порога tau in [1, 2, 3, 5, 8, 12, 20] пк (усереднення за кількома зернами), два графіки;
- Крок 12: Візуалізація узгоджених (зелених) та відкинутих (червоних) зіставлень на одному зображенні.
"""
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    imread_bgr, save_csv,
    extract_sift_features, apply_ratio_test,
    compute_reprojection_errors, custom_ransac_homography, homography_corner_distance
)


def n_iterations(p, w, s=4):
    """N = log(1 - p) / log(1 - (1 - w)^s), округлення вгору."""
    return int(np.ceil(np.log(1.0 - p) / np.log(1.0 - (1.0 - w) ** s)))


def run_part_c():
    print("=" * 60)
    print("Запуск Частини C: Власний алгоритм RANSAC (Кроки 8-12)")
    print("=" * 60)

    name_a, name_b = "pano_4", "pano_5"
    img_a = imread_bgr(os.path.join(DATA_DIR, f"{name_a}.jpg"))
    img_b = imread_bgr(os.path.join(DATA_DIR, f"{name_b}.jpg"))
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    kps_a, desc_a = extract_sift_features(gray_a)
    kps_b, desc_b = extract_sift_features(gray_b)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = bf.knnMatch(desc_a, desc_b, k=2)
    candidates = apply_ratio_test(matches_knn, ratio=0.80)
    n_cand = len(candidates)
    print(f"Вхідні дані ({name_a}-{name_b}): {n_cand} пар-кандидатів після тесту Лоу (T=0.80)")

    src_pts = np.float32([kps_a[m.queryIdx].pt for m in candidates]).reshape(-1, 2)
    dst_pts = np.float32([kps_b[m.trainIdx].pt for m in candidates]).reshape(-1, 2)

    tau0 = 3.0
    p = 0.99

    # -------------------------------------------------------------
    # Крок 9: Кількість ітерацій
    # -------------------------------------------------------------
    print("\n--- Крок 9: Дослідження стратегій кількості ітерацій N ---")
    N_theo_50 = n_iterations(p, 0.5)  # 72

    # Спостережувана частка викидів: «довгий» еталонний запуск N = 5000
    _, mask_ref, _, _ = custom_ransac_homography(src_pts, dst_pts, tau=tau0, p=p, max_iters=5000, mode="fixed", seed=0)
    n_ref = int(np.sum(mask_ref))
    w_obs = 1.0 - n_ref / n_cand
    N_theo_obs = n_iterations(p, w_obs)
    print(f"N для w=0.50: {N_theo_50};  спостережувана w = {w_obs:.3f} (inliers={n_ref}/{n_cand}) -> N = {N_theo_obs}")

    def p_success(N):
        return 1.0 - (1.0 - (1.0 - w_obs) ** 4) ** N

    modes_to_test = [
        (f"Теоретичне для w = 0.5 (N = {N_theo_50})", "fixed", N_theo_50),
        (f"Теоретичне для w = {w_obs:.3f} (N = {N_theo_obs})", "fixed", N_theo_obs),
        ("Фіксоване N = 1000", "fixed", 1000),
        ("Адаптивна зупинка (N_max = 2000)", "adaptive", 2000),
    ]

    n_runs = 30
    success_level = 0.95 * n_ref
    iter_rows = []
    for label, mode, iters in modes_to_test:
        times, inl, errs, its = [], [], [], []
        for r in range(n_runs):
            t0 = time.perf_counter()
            _, m, it, e = custom_ransac_homography(src_pts, dst_pts, tau=tau0, p=p, max_iters=iters, mode=mode, seed=100 + r)
            times.append((time.perf_counter() - t0) * 1000.0)
            inl.append(int(np.sum(m)))
            errs.append(e)
            its.append(it)
        inl = np.array(inl)
        succ = float(np.mean(inl >= success_level) * 100.0)
        p_theo = p_success(np.mean(its)) * 100.0
        print(f"  {label:42s}: N={np.mean(its):6.1f}, inliers={inl.mean():6.1f}±{inl.std():5.1f} "
              f"(min {inl.min()}), успіх={succ:5.1f}% (теор. {p_theo:5.1f}%), err={np.mean(errs):.2f}, t={np.mean(times):5.1f} мс")
        iter_rows.append([label, mode, iters, f"{np.mean(its):.1f}", f"{inl.mean():.1f}", f"{inl.std():.1f}",
                          int(inl.min()), f"{succ:.0f}", f"{p_theo:.0f}", f"{np.mean(errs):.2f}", f"{np.mean(times):.1f}"])

    save_csv(os.path.join(RESULTS_DIR, "c_ransac_iterations.csv"),
             ["strategy", "mode", "max_iters", "actual_iters_mean", "inliers_mean", "inliers_std", "inliers_min",
              "success_pct", "success_theory_pct", "mean_reprojection_err", "time_ms"],
             iter_rows)
    save_csv(os.path.join(RESULTS_DIR, "c_outlier_ratio.csv"),
             ["candidates", "reference_inliers", "w_observed", "N_w050", "N_w_observed", "n_runs"],
             [[n_cand, n_ref, f"{w_obs:.3f}", N_theo_50, N_theo_obs, n_runs]])

    # -------------------------------------------------------------
    # Крок 10: Порівняння з cv2.findHomography
    # -------------------------------------------------------------
    print("\n--- Крок 10: Звірка власного RANSAC з cv2.findHomography ---")
    t0 = time.perf_counter()
    H_custom, mask_custom, iters_custom, err_custom = custom_ransac_homography(
        src_pts, dst_pts, tau=tau0, p=p, max_iters=2000, mode="adaptive", seed=42)
    t_custom = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    H_cv, mask_cv = cv2.findHomography(src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, tau0)
    t_cv = (time.perf_counter() - t0) * 1000.0
    mask_cv = mask_cv.ravel() == 1

    inl_custom, inl_cv = int(mask_custom.sum()), int(mask_cv.sum())
    err_cv = float(np.mean(compute_reprojection_errors(src_pts[mask_cv], dst_pts[mask_cv], H_cv)))
    common = mask_custom & mask_cv
    n_common = int(common.sum())
    iou = n_common / int((mask_custom | mask_cv).sum()) * 100.0

    err_custom_common = float(np.mean(compute_reprojection_errors(src_pts[common], dst_pts[common], H_custom)))
    err_cv_common = float(np.mean(compute_reprojection_errors(src_pts[common], dst_pts[common], H_cv)))
    corner_diff = homography_corner_distance(H_custom, H_cv, gray_a.shape)
    pc = cv2.perspectiveTransform(src_pts[common].reshape(-1, 1, 2), H_custom).reshape(-1, 2)
    pv = cv2.perspectiveTransform(src_pts[common].reshape(-1, 1, 2), H_cv).reshape(-1, 2)
    point_diff = float(np.mean(np.linalg.norm(pc - pv, axis=1)))

    print(f"  Власний: inliers={inl_custom}, err={err_custom:.2f} пк, ітерацій={iters_custom}, t={t_custom:.1f} мс")
    print(f"  OpenCV : inliers={inl_cv}, err={err_cv:.2f} пк, t={t_cv:.1f} мс")
    print(f"  Спільних={n_common}, IoU={iou:.1f}%, err на спільних: {err_custom_common:.2f} / {err_cv_common:.2f} пк")
    print(f"  Розбіжність моделей: кути кадру {corner_diff:.2f} пк, точки inliers {point_diff:.2f} пк")

    save_csv(os.path.join(RESULTS_DIR, "c_ransac_comparison.csv"),
             ["implementation", "inliers", "reprojection_err", "err_on_common", "time_ms"],
             [["Власний RANSAC", inl_custom, f"{err_custom:.2f}", f"{err_custom_common:.2f}", f"{t_custom:.1f}"],
              ["OpenCV RANSAC", inl_cv, f"{err_cv:.2f}", f"{err_cv_common:.2f}", f"{t_cv:.1f}"]])
    save_csv(os.path.join(RESULTS_DIR, "c_ransac_agreement.csv"),
             ["common_inliers", "iou_pct", "corner_diff_px", "point_diff_px", "custom_iters"],
             [[n_common, f"{iou:.1f}", f"{corner_diff:.2f}", f"{point_diff:.2f}", iters_custom]])

    # -------------------------------------------------------------
    # Крок 11: Дослідження порога tau
    # -------------------------------------------------------------
    print("\n--- Крок 11: Дослідження порога tau ---")
    tau_grid = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    n_seeds = 10
    tau_rows = []
    inl_mean, inl_std, err_mean, err_std, inl_cv_list = [], [], [], [], []
    for tau in tau_grid:
        counts, errs = [], []
        for sd in range(n_seeds):
            _, m, _, e = custom_ransac_homography(src_pts, dst_pts, tau=tau, p=p, max_iters=2000, mode="fixed", seed=200 + sd)
            counts.append(int(m.sum()))
            errs.append(e)
        _, m_cv = cv2.findHomography(src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, tau,
                                     maxIters=5000, confidence=0.999)
        c_cv = int(m_cv.sum())
        counts, errs = np.array(counts), np.array(errs)
        inl_mean.append(counts.mean()); inl_std.append(counts.std())
        err_mean.append(errs.mean()); err_std.append(errs.std())
        inl_cv_list.append(c_cv)
        print(f"  tau={tau:4.1f}: inliers={counts.mean():6.1f}±{counts.std():4.1f} ({counts.mean()/n_cand*100:4.1f}%), "
              f"err={errs.mean():.2f}±{errs.std():.2f} пк, OpenCV inliers={c_cv}")
        tau_rows.append([f"{tau:.1f}", f"{counts.mean():.1f}", f"{counts.std():.1f}", f"{counts.mean()/n_cand*100:.1f}",
                         f"{errs.mean():.2f}", f"{errs.std():.2f}", c_cv])

    save_csv(os.path.join(RESULTS_DIR, "c_ransac_threshold.csv"),
             ["tau", "inliers_mean", "inliers_std", "inliers_pct", "err_mean", "err_std", "inliers_opencv"],
             tau_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.errorbar(tau_grid, inl_mean, yerr=inl_std, fmt="o-", color="#1b9e77", lw=2.0, ms=6, capsize=3,
                 label=f"Власний RANSAC (середнє ± σ, {n_seeds} запусків)")
    ax1.plot(tau_grid, inl_cv_list, "x--", color="#7570b3", lw=1.2, ms=7, label="cv2.findHomography (RANSAC)")
    ax1.axvspan(2.0, 5.0, color="#2ca02c", alpha=0.13, label="Плато стабільності (2–5 пк)")
    ax1.axvspan(8.0, 20.0, color="#d62728", alpha=0.10, label="Захоплення викидів (> 8 пк)")
    ax1.axvline(tau0, color="#d95f02", linestyle="--", lw=1.4, label="Робочий поріг τ = 3 пк")
    ax1.set_xlabel("Поріг узгодженості $\\tau$, пк")
    ax1.set_ylabel("Кількість узгоджених точок, од.")
    ax1.set_title("Кількість inliers від порога $\\tau$ (pano_4 – pano_5)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8)

    ax2.errorbar(tau_grid, err_mean, yerr=err_std, fmt="s-", color="#d95f02", lw=2.0, ms=6, capsize=3,
                 label=f"Власний RANSAC (середнє ± σ, {n_seeds} запусків)")
    ax2.axvspan(2.0, 5.0, color="#2ca02c", alpha=0.13, label="Плато стабільності (2–5 пк)")
    ax2.axvspan(8.0, 20.0, color="#d62728", alpha=0.10, label="Деградація точності (> 8 пк)")
    ax2.axvline(tau0, color="#1b9e77", linestyle="--", lw=1.4, label="Робочий поріг τ = 3 пк")
    ax2.set_xlabel("Поріг узгодженості $\\tau$, пк")
    ax2.set_ylabel("Середня похибка на inliers, пк")
    ax2.set_title("Середня похибка репроєкції на inliers від $\\tau$")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "c_threshold_analysis.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Крок 12: Візуалізація узгоджених та відкинутих зіставлень
    # -------------------------------------------------------------
    print("\n--- Крок 12: Візуалізація узгоджених та відкинутих пар ---")
    scale_disp = 0.65
    disp_a = cv2.resize(img_a, (0, 0), fx=scale_disp, fy=scale_disp, interpolation=cv2.INTER_AREA)
    disp_b = cv2.resize(img_b, (0, 0), fx=scale_disp, fy=scale_disp, interpolation=cv2.INTER_AREA)
    h_d, w_d = disp_a.shape[:2]
    vis = np.zeros((h_d, w_d * 2, 3), dtype=np.uint8)
    vis[:, :w_d] = disp_a
    vis[:, w_d:] = disp_b

    inlier_idx = np.where(mask_custom)[0]
    outlier_idx = np.where(~mask_custom)[0]
    rng = np.random.default_rng(42)
    s_out = rng.choice(outlier_idx, size=min(120, len(outlier_idx)), replace=False)
    s_in = rng.choice(inlier_idx, size=min(120, len(inlier_idx)), replace=False)

    def draw(idx_list, color, thick):
        for i in idx_list:
            p1 = (int(src_pts[i, 0] * scale_disp), int(src_pts[i, 1] * scale_disp))
            p2 = (int(dst_pts[i, 0] * scale_disp) + w_d, int(dst_pts[i, 1] * scale_disp))
            cv2.line(vis, p1, p2, color, thick, cv2.LINE_AA)
            cv2.circle(vis, p1, 3, color, -1)
            cv2.circle(vis, p2, 3, color, -1)

    draw(s_out, (0, 0, 230), 1)
    draw(s_in, (0, 200, 0), 1)
    cv2.imwrite(os.path.join(FIGURES_DIR, "c_inliers_outliers.png"), vis)

    with open(os.path.join(RESULTS_DIR, "c_log.txt"), "w", encoding="utf-8") as f:
        f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ C ===\n")
        f.write(f"Кандидатів ({name_a}-{name_b}): {n_cand}; еталонних inliers (N=5000): {n_ref}; w_obs = {w_obs:.3f}.\n")
        f.write(f"N(w=0.5) = {N_theo_50}; N(w_obs) = {N_theo_obs}.\n")
        f.write(f"Власний RANSAC: {inl_custom} inliers, {err_custom:.2f} пк; OpenCV: {inl_cv} inliers, {err_cv:.2f} пк.\n")
        f.write(f"Спільних: {n_common}, IoU = {iou:.1f}%; розбіжність кутів = {corner_diff:.2f} пк.\n")
        f.write(f"Візуалізація: {len(s_in)} з {len(inlier_idx)} inliers та {len(s_out)} з {len(outlier_idx)} outliers.\n")

    print("\nЧастина C успішно виконана!")


if __name__ == "__main__":
    run_part_c()
