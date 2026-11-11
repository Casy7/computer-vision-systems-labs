"""
Лабораторна робота № 4. Частина B: Чому потрібен робастний метод (Кроки 6-7).
- Крок 6: Оцінка гомографії звичайним методом найменших квадратів (МНК / OLS, cv2.findHomography(src, dst, 0));
- Крок 7: Застосування H_OLS через cv2.warpPerspective, візуалізація катастрофічного спотворення,
          обчислення середньої, медіанної та максимальної похибки репроєкції (порівняння з RANSAC).
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    imread_bgr, save_csv,
    extract_sift_features, apply_ratio_test, compute_reprojection_errors
)


def run_part_b():
    print("=" * 60)
    print("Запуск Частини B: Дослідження чутливості МНК до викидів (Кроки 6-7)")
    print("=" * 60)

    pairs = [
        ("pano_1", "pano_2", os.path.join(DATA_DIR, "pano_1.jpg"), os.path.join(DATA_DIR, "pano_2.jpg")),
        ("pano_4", "pano_5", os.path.join(DATA_DIR, "pano_4.jpg"), os.path.join(DATA_DIR, "pano_5.jpg"))
    ]

    csv_rows = []

    for name_a, name_b, path_a, path_b in pairs:
        print(f"\n--- Аналіз пари {name_a} -> {name_b} ---")
        img_a = imread_bgr(path_a)
        img_b = imread_bgr(path_b)
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        kps_a, desc_a = extract_sift_features(gray_a)
        kps_b, desc_b = extract_sift_features(gray_b)

        # k-NN зіставлення та вибірка кандидатів при T = 0.80 (реалістична присутність викидів)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches_knn = bf.knnMatch(desc_a, desc_b, k=2)
        candidates = apply_ratio_test(matches_knn, ratio=0.80)
        print(f"Відібрано кандидатів Лоу (T=0.80): {len(candidates)}")

        src_pts = np.float32([kps_a[m.queryIdx].pt for m in candidates]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps_b[m.trainIdx].pt for m in candidates]).reshape(-1, 1, 2)

        # Крок 6: Оцінка гомографії класичним МНК (метод 0 - без відсіву викидів)
        H_ols, mask_ols = cv2.findHomography(src_pts, dst_pts, 0)
        errors_ols = compute_reprojection_errors(src_pts, dst_pts, H_ols)
        mean_err_ols = float(np.mean(errors_ols))
        median_err_ols = float(np.median(errors_ols))
        max_err_ols = float(np.max(errors_ols))
        print(f"МНК (OLS, без відсіву): похибка репроєкції mean={mean_err_ols:.2f} пк, median={median_err_ols:.2f} пк, max={max_err_ols:.2f} пк")

        # Оцінка гомографії через RANSAC (поріг 3.0 пк) для еталонного порівняння
        H_ransac, mask_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        inlier_mask = (mask_ransac.ravel() == 1)
        errors_ransac_all = compute_reprojection_errors(src_pts, dst_pts, H_ransac)
        errors_ransac_inliers = errors_ransac_all[inlier_mask]

        num_inliers = int(np.sum(inlier_mask))
        num_outliers = len(candidates) - num_inliers
        outlier_ratio = (num_outliers / len(candidates)) * 100.0

        mean_err_ransac = float(np.mean(errors_ransac_inliers))
        median_err_ransac = float(np.median(errors_ransac_inliers))
        max_err_ransac = float(np.max(errors_ransac_inliers))
        mean_err_ols_on_inl = float(np.mean(errors_ols[inlier_mask]))
        median_err_ols_on_inl = float(np.median(errors_ols[inlier_mask]))
        print(f"RANSAC (tau=3.0 пк): inliers={num_inliers}/{len(candidates)} (викиди: {outlier_ratio:.1f}%), "
              f"похибка на inliers: mean={mean_err_ransac:.2f} пк, median={median_err_ransac:.2f} пк, max={max_err_ransac:.2f} пк")
        print(f"МНК на тих самих inliers: mean={mean_err_ols_on_inl:.1f} пк, median={median_err_ols_on_inl:.1f} пк")

        csv_rows.append([
            f"{name_a}-{name_b}",
            len(candidates),
            num_inliers,
            num_outliers,
            f"{outlier_ratio:.1f}",
            f"{mean_err_ols:.2f}",
            f"{median_err_ols:.2f}",
            f"{max_err_ols:.2f}",
            f"{mean_err_ols_on_inl:.2f}",
            f"{median_err_ols_on_inl:.2f}",
            f"{mean_err_ransac:.2f}",
            f"{median_err_ransac:.2f}",
            f"{max_err_ransac:.2f}"
        ])

        # Крок 7: Візуалізація результатів для pano_1 -> pano_2
        if name_a == "pano_1":
            print("\nГенерація порівняльних ілюстрацій варпінгу (Крок 7)...")
            h, w = img_b.shape[:2]

            warped_ols = cv2.warpPerspective(img_a, H_ols, (w, h), flags=cv2.INTER_LINEAR)
            warped_ransac = cv2.warpPerspective(img_a, H_ransac, (w, h), flags=cv2.INTER_LINEAR)

            alpha = 0.5
            overlay_ols = cv2.addWeighted(img_b, alpha, warped_ols, 1.0 - alpha, 0)
            overlay_ransac = cv2.addWeighted(img_b, alpha, warped_ransac, 1.0 - alpha, 0)

            cv2.imwrite(os.path.join(FIGURES_DIR, "b_warped_ols.png"), warped_ols)
            cv2.imwrite(os.path.join(FIGURES_DIR, "b_warped_ransac.png"), warped_ransac)
            cv2.imwrite(os.path.join(FIGURES_DIR, "b_overlay_ols.png"), overlay_ols)
            cv2.imwrite(os.path.join(FIGURES_DIR, "b_overlay_ransac.png"), overlay_ransac)

            fig, axs = plt.subplots(2, 2, figsize=(10, 6.2))

            axs[0, 0].imshow(cv2.cvtColor(warped_ols, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title(f"(а) Варпінг МНК (без відсіву): mean e = {mean_err_ols:.1f} пк", fontsize=9.5)
            axs[0, 0].axis("off")

            axs[0, 1].imshow(cv2.cvtColor(overlay_ols, cv2.COLOR_BGR2RGB))
            axs[0, 1].set_title("(б) Накладання МНК на цільовий кадр pano_2", fontsize=9.5)
            axs[0, 1].axis("off")

            axs[1, 0].imshow(cv2.cvtColor(warped_ransac, cv2.COLOR_BGR2RGB))
            axs[1, 0].set_title(f"(в) Варпінг RANSAC (τ = 3 пк): mean e на inliers = {mean_err_ransac:.2f} пк", fontsize=9.5)
            axs[1, 0].axis("off")

            axs[1, 1].imshow(cv2.cvtColor(overlay_ransac, cv2.COLOR_BGR2RGB))
            axs[1, 1].set_title("(г) Накладання RANSAC на цільовий кадр pano_2", fontsize=9.5)
            axs[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, "b_ols_vs_ransac_panel.png"))
            plt.close(fig)

            fig_hist, ax_h = plt.subplots(figsize=(6.8, 3.9))
            all_err = np.concatenate([errors_ols, errors_ransac_inliers])
            lo = np.floor(np.log10(max(all_err.min(), 1e-2)))
            hi = np.ceil(np.log10(all_err.max()))
            bins = np.logspace(lo, hi, int((hi - lo) * 10) + 1)
            ax_h.hist(errors_ols, bins=bins, color="#d95f02", alpha=0.45,
                      label=f"МНК, усі {len(errors_ols)} кандидатів: mean = {mean_err_ols:.1f} пк")
            ax_h.hist(errors_ols[inlier_mask], bins=bins, color="#7570b3", alpha=0.75, histtype="stepfilled",
                      label=f"МНК, лише {num_inliers} inliers RANSAC: mean = {mean_err_ols_on_inl:.1f} пк")
            ax_h.hist(errors_ransac_inliers, bins=bins, color="#1b9e77", alpha=0.8,
                      label=f"RANSAC, {num_inliers} inliers: mean = {mean_err_ransac:.2f} пк")
            ax_h.axvline(3.0, color="k", ls="--", lw=1.0)
            ax_h.text(3.3, ax_h.get_ylim()[1] * 0.90, "τ = 3 пк", fontsize=8)
            ax_h.set_xscale("log")
            ax_h.set_xlabel("Похибка репроєкції $e_i = \\|p_2 - \\pi(H p_1)\\|$, пк (лог. шкала)")
            ax_h.set_ylabel("Кількість пар точок, од.")
            ax_h.set_title("Розподіл похибок репроєкції: МНК проти RANSAC (pano_1 – pano_2)")
            ax_h.grid(True, which="both", linestyle=":", alpha=0.5)
            ax_h.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, "b_reprojection_hist.png"))
            plt.close(fig_hist)

    header = [
        "pair", "candidates", "inliers_ransac", "outliers_ransac", "outlier_pct",
        "ols_mean_err", "ols_median_err", "ols_max_err",
        "ols_mean_err_on_inliers", "ols_median_err_on_inliers",
        "ransac_mean_err", "ransac_median_err", "ransac_max_err"
    ]
    save_csv(os.path.join(RESULTS_DIR, "b_ols_vs_ransac.csv"), header, csv_rows)

    with open(os.path.join(RESULTS_DIR, "b_log.txt"), "w", encoding="utf-8") as f:
        f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ B ===\n")
        f.write("Крок 6: Оцінено гомографію МНК без фільтрації викидів (cv2.findHomography(src, dst, 0)).\n")
        f.write("Крок 7: Застосовано warpPerspective. Через чутливість МНК до викидів геометрія повністю руйнується.\n")
        f.write("Збережено порівняльні рисунки та гістограму похибок репроєкції для pano_1-pano_2.\n")

    print("\nЧастина B успішно завершена!")


if __name__ == "__main__":
    run_part_b()
