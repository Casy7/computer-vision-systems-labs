"""
Лабораторна робота № 3. Частина B.
Чутливість кутового детектора Гарріса до масштабу (Кроки 9–10).
"""
import os
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from common import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    imread_bgr, imread_gray,
    detect_custom_harris, detect_sift_keypoints, save_csv
)

print("=== Частина B: Чутливість детектора Гарріса до зміни масштабу ===")

image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))

scale_results_rows = []

base_path = os.path.join(DATA_DIR, "paintings.jpg")
base_img_bgr = imread_bgr(base_path)
base_rgb = cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2RGB)
base_gray = cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2GRAY)
h, w = base_gray.shape

# Детекція на оригіналі
pts_orig, _, _ = detect_custom_harris(base_gray, max_points=1000, radius=10)

# Зменшення в 2 рази з антиаліасингом INTER_AREA (Крок 9)
base_gray_half = cv2.resize(base_gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
pts_half, _, _ = detect_custom_harris(base_gray_half, max_points=1000, radius=10)

# Перенесення в координати оригіналу (Крок 10: множення на 2)
pts_half_to_orig = pts_half * 2.0

# Розрахунок метрики відповідності ЗА ФОРМУЛЮВАННЯМ КРОКУ 10:
# "яка частка точок оригіналу має відповідника в межах 3 пікселів серед перенесених точок"
tree_half = cKDTree(pts_half_to_orig)
dists_o2h, _ = tree_half.query(pts_orig)

match_2px = float(np.mean(dists_o2h < 2.0) * 100.0)
match_3px = float(np.mean(dists_o2h < 3.0) * 100.0)
match_5px = float(np.mean(dists_o2h < 5.0) * 100.0)
median_dist = float(np.median(dists_o2h))

print(f"Базове зображення: {os.path.basename(base_path)}")
print(f"  Оригінал (N={len(pts_orig)}), Зменшене x0.5 (N={len(pts_half)})")
print(f"  Частка точок оригіналу з відповідником у межах < 2 пк: {match_2px:.2f}%")
print(f"  Частка точок оригіналу з відповідником у межах < 3 пк: {match_3px:.2f}% (демонстрація неінваріантності)")
print(f"  Частка точок оригіналу з відповідником у межах < 5 пк: {match_5px:.2f}%")
print(f"  Медіанна відстань від точок оригіналу до найближчого кута: {median_dist:.2f} пк")

# Масштабно-інваріантний SIFT (з відсіканням наддрібного шуму першої октави)
pts_sift_orig, _, _ = detect_sift_keypoints(base_gray, max_points=1000, radius=10, min_size=5.0)
pts_sift_half, _, _ = detect_sift_keypoints(base_gray_half, max_points=1000, radius=5, min_size=2.5)
pts_sift_h2o = pts_sift_half * 2.0
tree_sift_half = cKDTree(pts_sift_h2o)
d_sift, _ = tree_sift_half.query(pts_sift_orig)
sift_match_3px = float(np.mean(d_sift < 3.0) * 100.0)
print(f"  SIFT на тій же парі кадру дає збіг < 3 пк: {sift_match_3px:.2f}% (завдяки піраміді масштабів DoG)")

# Обробка всіх 4 зображень
for p in image_paths:
    fname = os.path.basename(p)
    g = imread_gray(p)
    gh, gw = g.shape
    g_half = cv2.resize(g, (gw // 2, gh // 2), interpolation=cv2.INTER_AREA)

    p_orig, _, _ = detect_custom_harris(g, max_points=1000, radius=10)
    p_half, _, _ = detect_custom_harris(g_half, max_points=1000, radius=10)
    p_mapped = p_half * 2.0

    t_half = cKDTree(p_mapped)
    d, _ = t_half.query(p_orig)

    m2 = float(np.mean(d < 2.0) * 100.0)
    m3 = float(np.mean(d < 3.0) * 100.0)
    m5 = float(np.mean(d < 5.0) * 100.0)
    med = float(np.median(d))

    # SIFT
    ps_orig, _, _ = detect_sift_keypoints(g, max_points=1000, radius=10, min_size=5.0)
    ps_half, _, _ = detect_sift_keypoints(g_half, max_points=1000, radius=5, min_size=2.5)
    ps_mapped = ps_half * 2.0
    ts_half = cKDTree(ps_mapped)
    ds, _ = ts_half.query(ps_orig)
    m3_sift = float(np.mean(ds < 3.0) * 100.0)

    scale_results_rows.append([
        fname,
        len(p_orig),
        len(p_half),
        f"{m2:.2f}",
        f"{m3:.2f}",
        f"{m5:.2f}",
        f"{med:.2f}",
        f"{m3_sift:.2f}"
    ])

save_csv(
    os.path.join(RESULTS_DIR, "b_scale_sensitivity.csv"),
    [
        "Image", "Points_Orig", "Points_Half",
        "Harris_Match_lt2px_pct", "Harris_Match_lt3px_pct", "Harris_Match_lt5px_pct",
        "Harris_Median_Dist_px", "SIFT_Match_lt3px_pct"
    ],
    scale_results_rows
)
print("Збережено b_scale_sensitivity.csv")

# =========================================================================
# Побудова візуалізацій
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

cy, cx = h // 2, w // 2
c_y1, c_y2 = cy - 250, cy + 250
c_x1, c_x2 = cx - 350, cx + 350

crop_orig = base_rgb[c_y1:c_y2, c_x1:c_x2].copy()

mask_o = (
    (pts_orig[:, 0] >= c_x1) & (pts_orig[:, 0] < c_x2) &
    (pts_orig[:, 1] >= c_y1) & (pts_orig[:, 1] < c_y2)
)
pts_o_crop = pts_orig[mask_o]

mask_h = (
    (pts_half_to_orig[:, 0] >= c_x1) & (pts_half_to_orig[:, 0] < c_x2) &
    (pts_half_to_orig[:, 1] >= c_y1) & (pts_half_to_orig[:, 1] < c_y2)
)
pts_h_crop = pts_half_to_orig[mask_h]

for pt in pts_o_crop:
    cv2.circle(crop_orig, (int(pt[0] - c_x1), int(pt[1] - c_y1)), 4, (255, 30, 30), -1)

for pt in pts_h_crop:
    cv2.circle(crop_orig, (int(pt[0] - c_x1), int(pt[1] - c_y1)), 7, (30, 220, 30), 2)

axes[0].imshow(crop_orig)
axes[0].plot([], [], "o", color=(1.0, 0.12, 0.12), label=f"Кути оригіналу $I_1$ ($N={len(pts_o_crop)}$)")
axes[0].plot([], [], "s", markerfacecolor="none", markeredgecolor=(0.12, 0.86, 0.12), markeredgewidth=2, label=f"Кути $I_2 \\times 2$ ($N={len(pts_h_crop)}$)")
axes[0].set_title("(а) Неузгодженість кутів Гарріса при зміні масштабу $\\times 2$", fontsize=10)
axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9)
axes[0].axis("off")

# Гістограма зміщень
axes[1].hist(dists_o2h, bins=35, range=(0, 35), color="#d95f02", edgecolor="black", alpha=0.75, density=False)
axes[1].axvline(3.0, color="red", linestyle="--", linewidth=1.5, label=f"Поріг $\\varepsilon=3$ пк (збіг лише {match_3px:.1f}%)")
axes[1].axvline(median_dist, color="blue", linestyle=":", linewidth=1.5, label=f"Медіана нев'язки ({median_dist:.1f} пк)")
axes[1].set_xlabel("Відстань від точки оригіналу до найближчого кута $I_2 \\times 2$, пікселі", fontsize=9)
axes[1].set_ylabel("Кількість кутів", fontsize=9)
axes[1].set_title(r"(б) Розподіл відстаней між точками оригіналу та $I_2 \times 2$", fontsize=10)
axes[1].legend(loc="upper right", framealpha=0.9, fontsize=9)
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "b_scale_comparison.png"))
plt.close(fig)
print("Збережено b_scale_comparison.png")

# Журнал виконання Частини B
log_path = os.path.join(RESULTS_DIR, "b_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ B: ЧУТЛИВІСТЬ ДО МАСШТАБУ ===\n")
    f.write(f"Базове зображення: {os.path.basename(base_path)}\n")
    f.write(f"Роздільність оригіналу: {w} x {h}\n")
    f.write(f"Роздільність зменшеного кадру: {w // 2} x {h // 2}\n")
    f.write(f"Кутів на оригіналі: {len(pts_orig)}, на зменшеному: {len(pts_half)}\n")
    f.write(f"Частка точок оригіналу з відповідником у межах 3 пк: {match_3px:.2f}%\n")
    f.write(f"Медіанне відхилення координат: {median_dist:.2f} пк\n")
    f.write(f"SIFT (з пірамідою DoG) на тій же парі: {sift_match_3px:.2f}%\n")

print(f"Збережено {log_path}")
print("Частина B успішно виконана!\n")
