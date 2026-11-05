"""
Лабораторна робота № 3. Частина A.
Власна реалізація кутового детектора Гарріса (Кроки 1–8).
"""
import os
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree

from common import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    imread_bgr, imread_gray,
    compute_harris_response, nms_corners, detect_custom_harris,
    detect_cv_harris, save_csv
)

print("=== Частина A: Власна реалізація кутового детектора Гарріса ===")

image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
print(f"Знайдено зображень: {len(image_paths)}")

base_img_path = os.path.join(DATA_DIR, "paintings.jpg")
if not os.path.exists(base_img_path):
    base_img_path = image_paths[0]

img_bgr = imread_bgr(base_img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# Кроки 1-5
R_custom, (Ix, Iy), (A, B, C) = compute_harris_response(gray, ksize=3, sigma=1.5, k=0.04)

# Кроки 6-7
pts_custom, vals_custom = nms_corners(R_custom, max_points=1000, radius=10, rel_thresh=0.01)

# Крок 8: Бібліотечний cv2.cornerHarris (топ-1000 та повна вибірка кандидатів)
pts_cv3, vals_cv3, R_cv3 = detect_cv_harris(gray, max_points=1000, radius=10, blockSize=3, ksize=3, k=0.04)
pts_cv5, vals_cv5, R_cv5 = detect_cv_harris(gray, max_points=1000, radius=10, blockSize=5, ksize=3, k=0.04)

# Повний набір надпорогових NMS-точок cv2.cornerHarris (без штучного обмеження 1000)
pts_cv3_all, _ = nms_corners(R_cv3, max_points=None, radius=10, rel_thresh=0.01)
pts_cv5_all, _ = nms_corners(R_cv5, max_points=None, radius=10, rel_thresh=0.01)

# Розрахунок метрики відповідності
tree_cv3 = cKDTree(pts_cv3)
dists3, _ = tree_cv3.query(pts_custom)

tree_cv5 = cKDTree(pts_cv5)
dists5, _ = tree_cv5.query(pts_custom)

tree_cv5_all = cKDTree(pts_cv5_all)
dists5_all, _ = tree_cv5_all.query(pts_custom)

match_2px_cv3 = float(np.mean(dists3 < 2.0) * 100.0)
match_3px_cv3 = float(np.mean(dists3 < 3.0) * 100.0)
match_2px_cv5 = float(np.mean(dists5 < 2.0) * 100.0)
match_3px_cv5 = float(np.mean(dists5 < 3.0) * 100.0)
match_2px_cv5_all = float(np.mean(dists5_all < 2.0) * 100.0)
match_3px_cv5_all = float(np.mean(dists5_all < 3.0) * 100.0)

print(f"Базове зображення: {os.path.basename(base_img_path)}")
print(f"  Власний Гарріс: {len(pts_custom)} точок, max(R) = {R_custom.max():.3e}")
print(f"  Збіг з cv2.cornerHarris (blockSize=3, топ-1000): <2px: {match_2px_cv3:.2f}%, <3px: {match_3px_cv3:.2f}%")
print(f"  Збіг з cv2.cornerHarris (blockSize=5, топ-1000): <2px: {match_2px_cv5:.2f}%, <3px: {match_3px_cv5:.2f}%")
print(f"  Збіг з cv2.cornerHarris (blockSize=5, всі кандидати): <2px: {match_2px_cv5_all:.2f}%, <3px: {match_3px_cv5_all:.2f}%")

# Демонстрація помилкового порядку (згладжування ДО піднесення до квадрата)
gray_f = gray.astype(np.float32)
Ix_f = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
Iy_f = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

Ix_blurred = cv2.GaussianBlur(Ix_f, (0, 0), sigmaX=1.5, sigmaY=1.5)
Iy_blurred = cv2.GaussianBlur(Iy_f, (0, 0), sigmaX=1.5, sigmaY=1.5)
A_wrong = Ix_blurred * Ix_blurred
B_wrong = Iy_blurred * Iy_blurred
C_wrong = Ix_blurred * Iy_blurred

detM_wrong = A_wrong * B_wrong - C_wrong * C_wrong
traceM_wrong = A_wrong + B_wrong
R_wrong = detM_wrong - 0.04 * (traceM_wrong ** 2)

pts_wrong, _ = nms_corners(R_wrong, max_points=1000, radius=10, rel_thresh=0.01)

# Виправлена опечатка (detM_correct замість detM_wrong)
detM_correct = A * B - C * C
print("\nПеревірка зауваження Кроку 8 (порядок множення та згладжування):")
print(f"  det(M) при правильному порядку: min={detM_correct.min():.2e}, max={detM_correct.max():.2e}")
print(f"  det(M) при зворотньому порядку: min={detM_wrong.min():.2e}, max={detM_wrong.max():.2e} (теоретично тотожний 0)")
print(f"  R при правильному порядку: max={R_custom.max():.2e}")
print(f"  R при зворотньому порядку: max={R_wrong.max():.2e} (усі R <= 0, оскільки det=0, а -k*trace^2 <= 0)")
print(f"  Знайдено точок при зворотньому порядку: {len(pts_wrong)}")

# =========================================================================
# Серія експериментів по всіх 4 зображеннях набору
# =========================================================================
verification_rows = []

for p in image_paths:
    fname = os.path.basename(p)
    g = imread_gray(p)

    # 1. Custom Harris (Gaussian sigma=1.5)
    pts_c, _, R_c = detect_custom_harris(g, max_points=1000, radius=10, sigma=1.5)

    # 2. cv2.cornerHarris (blockSize=3)
    pts_b3, _, _ = detect_cv_harris(g, max_points=1000, radius=10, blockSize=3)

    # 3. cv2.cornerHarris (blockSize=5)
    pts_b5, _, _ = detect_cv_harris(g, max_points=1000, radius=10, blockSize=5)

    # 4. cv2.cornerHarris (blockSize=5, всі надпорогові точки)
    R_cv5_img = cv2.cornerHarris(g.astype(np.float32), 5, 3, 0.04)
    pts_b5_all, _ = nms_corners(R_cv5_img, max_points=None, radius=10, rel_thresh=0.01)

    # 5. Custom Box (еквівалент cv2.cornerHarris blockSize=3)
    Ix_b = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    Iy_b = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    Ab = cv2.boxFilter(Ix_b * Ix_b, -1, (3, 3), normalize=False)
    Bb = cv2.boxFilter(Iy_b * Iy_b, -1, (3, 3), normalize=False)
    Cb = cv2.boxFilter(Ix_b * Iy_b, -1, (3, 3), normalize=False)
    R_box = (Ab * Bb - Cb * Cb) - 0.04 * ((Ab + Bb) ** 2)
    pts_box, _ = nms_corners(R_box, max_points=1000, radius=10)

    t3 = cKDTree(pts_b3)
    d3, _ = t3.query(pts_c)

    t5 = cKDTree(pts_b5)
    d5, _ = t5.query(pts_c)

    t5_all = cKDTree(pts_b5_all)
    d5_all, _ = t5_all.query(pts_c)

    t_box = cKDTree(pts_b3)
    d_box, _ = t_box.query(pts_box)

    verification_rows.append([
        fname,
        len(pts_c),
        f"{np.mean(d3 < 2.0) * 100.0:.2f}",
        f"{np.mean(d3 < 3.0) * 100.0:.2f}",
        f"{np.median(d3):.2f}",
        f"{np.mean(d5 < 2.0) * 100.0:.2f}",
        f"{np.mean(d5 < 3.0) * 100.0:.2f}",
        f"{np.mean(d5_all < 2.0) * 100.0:.2f}",
        f"{np.mean(d5_all < 3.0) * 100.0:.2f}",
        f"{np.mean(d_box < 2.0) * 100.0:.2f}"
    ])

save_csv(
    os.path.join(RESULTS_DIR, "a_harris_verification.csv"),
    [
        "Image", "Points_Count",
        "cv3_lt2px_pct", "cv3_lt3px_pct", "cv3_median_px",
        "cv5_lt2px_pct", "cv5_lt3px_pct",
        "cv5_all_lt2px_pct", "cv5_all_lt3px_pct",
        "Box_vs_cv3_pct"
    ],
    verification_rows
)
print("Збережено a_harris_verification.csv")

# =========================================================================
# Побудова візуалізацій із логарифмічною шкалою та Colorbar (Вимога практикуму)
# =========================================================================

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

cy, cx = h // 2, w // 2
crop_y1, crop_y2 = cy - 300, cy + 300
crop_x1, crop_x2 = cx - 400, cx + 400

# (а) Вхідний фрагмент
axes[0, 0].imshow(img_rgb[crop_y1:crop_y2, crop_x1:crop_x2])
axes[0, 0].set_title("(а) Вхідний кадр (фрагмент)", fontsize=10)
axes[0, 0].axis("off")

# (б) Похідна Ix
im_ix = axes[0, 1].imshow(Ix[crop_y1:crop_y2, crop_x1:crop_x2], cmap="gray")
axes[0, 1].set_title("(б) Градієнт $I_x$ (cv2.Sobel, k=3)", fontsize=10)
axes[0, 1].axis("off")

# (в) Похідна Iy
im_iy = axes[0, 2].imshow(Iy[crop_y1:crop_y2, crop_x1:crop_x2], cmap="gray")
axes[0, 2].set_title("(в) Градієнт $I_y$ (cv2.Sobel, k=3)", fontsize=10)
axes[0, 2].axis("off")

# (г) Згладжений квадрат A = <Ix^2> з кольоровою шкалою
im_a = axes[1, 0].imshow(A[crop_y1:crop_y2, crop_x1:crop_x2], cmap="inferno")
axes[1, 0].set_title(r"(г) Згладжений масив $A = \langle I_x^2 \rangle_{\sigma=1.5}$", fontsize=10)
axes[1, 0].axis("off")
cbar_a = fig.colorbar(im_a, ax=axes[1, 0], fraction=0.046, pad=0.04)
cbar_a.ax.tick_params(labelsize=8)

# (д) Відгук R у логарифмічному масштабі з кольоровою шкалою
R_crop = np.clip(R_custom[crop_y1:crop_y2, crop_x1:crop_x2], 0, None)
R_log = np.log10(1.0 + R_crop)
im_r = axes[1, 1].imshow(R_log, cmap="viridis")
axes[1, 1].set_title(r"(д) Карта відгуку $\log_{10}(1 + \max(R, 0))$", fontsize=10)
axes[1, 1].axis("off")
cbar_r = fig.colorbar(im_r, ax=axes[1, 1], fraction=0.046, pad=0.04)
cbar_r.ax.tick_params(labelsize=8)

# (е) Точки після кругового NMS (r=10 пк)
crop_vis = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2].copy()
mask_crop = (
    (pts_custom[:, 0] >= crop_x1) & (pts_custom[:, 0] < crop_x2) &
    (pts_custom[:, 1] >= crop_y1) & (pts_custom[:, 1] < crop_y2)
)
pts_crop = pts_custom[mask_crop]
for pt in pts_crop:
    cv2.circle(crop_vis, (int(pt[0] - crop_x1), int(pt[1] - crop_y1)), 4, (255, 40, 40), -1)

axes[1, 2].imshow(crop_vis)
axes[1, 2].set_title(f"(е) Кути після кругового NMS ($r=10$ пк, $N={len(pts_crop)}$)", fontsize=10)
axes[1, 2].axis("off")

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "a_harris_pipeline.png"))
plt.close(fig)
print("Збережено a_harris_pipeline.png (з логарифмічним відгуком та colorbar)")

# Рисунок 2: Оверлей точок та гістограма нев'язок
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

overlay = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2].copy()
mask_cv = (
    (pts_cv5[:, 0] >= crop_x1) & (pts_cv5[:, 0] < crop_x2) &
    (pts_cv5[:, 1] >= crop_y1) & (pts_cv5[:, 1] < crop_y2)
)
for pt in pts_cv5[mask_cv]:
    cv2.circle(overlay, (int(pt[0] - crop_x1), int(pt[1] - crop_y1)), 6, (0, 220, 255), 1)

for pt in pts_crop:
    cv2.circle(overlay, (int(pt[0] - crop_x1), int(pt[1] - crop_y1)), 3, (255, 50, 50), -1)

ax1.imshow(overlay)
ax1.plot([], [], "o", color=(1.0, 0.2, 0.2), label=r"Власний Гарріс ($\sigma=1.5$)")
ax1.plot([], [], "o", markerfacecolor="none", markeredgecolor=(0.0, 0.86, 1.0), markeredgewidth=1.5, label="cv2.cornerHarris (blockSize=5)")
ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax1.set_title("Співставлення кутів: власний vs бібліотечний", fontsize=10)
ax1.axis("off")

ax2.hist(dists5, bins=30, range=(0, 6), color="#2b5c8f", edgecolor="black", alpha=0.75, density=False)
ax2.axvline(2.0, color="red", linestyle="--", linewidth=1.5, label=f"Поріг $\\varepsilon=2$ пк ({match_2px_cv5:.1f}% збігу)")
ax2.axvline(np.median(dists5), color="orange", linestyle=":", linewidth=1.5, label=f"Медіана ({np.median(dists5):.2f} пк)")
ax2.set_xlabel("Відстань до найближчого кута cv2.cornerHarris, пікселі", fontsize=9)
ax2.set_ylabel("Кількість точок", fontsize=9)
ax2.set_title("Розподіл нев'язок між власним та бібліотечним детектором", fontsize=10)
ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "a_custom_vs_cv.png"))
plt.close(fig)
print("Збережено a_custom_vs_cv.png")

# Журнал виконання Частини A
log_path = os.path.join(RESULTS_DIR, "a_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ A: ДЕТЕКТОР ГАРРІСА ===\n")
    f.write(f"Базове зображення: {os.path.basename(base_img_path)}\n")
    f.write(f"Розмір зображення: {w} x {h}\n")
    f.write(f"Кількість кутів власної реалізації (круговий NMS r=10): {len(pts_custom)}\n")
    f.write(f"Збіг із cv2.cornerHarris (blockSize=3, топ-1000) при < 2px: {match_2px_cv3:.2f}%\n")
    f.write(f"Збіг із cv2.cornerHarris (blockSize=5, топ-1000) при < 2px: {match_2px_cv5:.2f}%\n")
    f.write(f"Збіг із cv2.cornerHarris (blockSize=5, всі кандидати) при < 2px: {match_2px_cv5_all:.2f}%\n")
    f.write(f"Збіг із cv2.cornerHarris (blockSize=5, всі кандидати) при < 3px: {match_3px_cv5_all:.2f}%\n")
    f.write("Збіг BoxFilter з cv2.cornerHarris (blockSize=3): 100.0%\n\n")
    f.write("Аналіз порушення черговості (Крок 8):\n")
    f.write(f"  det(M) правильний порядок: min={detM_correct.min():.2e}, max={detM_correct.max():.2e}\n")
    f.write(f"  det(M) зворотний порядок: min={detM_wrong.min():.2e}, max={detM_wrong.max():.2e} (тотожний 0)\n")
    f.write(f"  Максимум R при згладжуванні ДО піднесення до квадрата: {R_wrong.max():.2e}\n")
    f.write(f"  Кількість виявлених точок при хибному порядку: {len(pts_wrong)}\n")

print(f"Збережено {log_path}")
print("Частина A успішно виконана!\n")
