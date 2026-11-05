"""
Лабораторна робота № 3. Частина C.
Дослідження детектора та дескриптора SIFT і ORB (Кроки 11–13).
"""
import os
import sys
import glob
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    imread_bgr, imread_gray, imread_rgb, save_csv
)

print("=== Частина C: Дослідження SIFT та ORB ===")

image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))

# Крок 11: Створення детекторів
sift_unlim = cv2.SIFT_create()
sift_1000 = cv2.SIFT_create(nfeatures=1000)
orb_1000 = cv2.ORB_create(nfeatures=1000)

comparison_rows = []

# Детальний аналіз на базовому кадрі paintings.jpg
base_path = os.path.join(DATA_DIR, "paintings.jpg")
base_bgr = imread_bgr(base_path)
base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
h, w = base_gray.shape

# Прогрів для точного таймінгу
_ = sift_unlim.detectAndCompute(base_gray, None)
_ = orb_1000.detectAndCompute(base_gray, None)

# Вимірювання часу (по 5 повторень)
n_runs = 5

# SIFT (без обмеження кількості)
t_sift_list = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    kps_sift_all, des_sift_all = sift_unlim.detectAndCompute(base_gray, None)
    t_sift_list.append((time.perf_counter() - t0) * 1000.0)
t_sift_all_ms = float(np.mean(t_sift_list))

# SIFT (обмеження 1000 точок)
t_sift1k_list = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    kps_sift1k, des_sift1k = sift_1000.detectAndCompute(base_gray, None)
    t_sift1k_list.append((time.perf_counter() - t0) * 1000.0)
t_sift_1k_ms = float(np.mean(t_sift1k_list))

# ORB (1000 точок)
t_orb_list = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    kps_orb1k, des_orb1k = orb_1000.detectAndCompute(base_gray, None)
    t_orb_list.append((time.perf_counter() - t0) * 1000.0)
t_orb_1k_ms = float(np.mean(t_orb_list))

print(f"Базове зображення: paintings.jpg ({w} x {h})")
print(f"  SIFT (unlim): {len(kps_sift_all)} точок, дескриптор: {des_sift_all.shape}, тип: {des_sift_all.dtype}, час: {t_sift_all_ms:.1f} мс")
print(f"  SIFT (1000):  {len(kps_sift1k)} точок, дескриптор: {des_sift1k.shape}, тип: {des_sift1k.dtype}, час: {t_sift_1k_ms:.1f} мс")
print(f"  ORB (1000):   {len(kps_orb1k)} точок, дескриптор: {des_orb1k.shape}, тип: {des_orb1k.dtype}, час: {t_orb_1k_ms:.1f} мс")
print(f"  Прискорення ORB порівняно з SIFT (1000 точок): {t_sift_1k_ms / t_orb_1k_ms:.1f}x")

# Вимірювання по всіх кадрах
for p in image_paths:
    fname = os.path.basename(p)
    g = imread_gray(p)
    
    # SIFT full
    t0 = time.perf_counter()
    ks_full, ds_full = sift_unlim.detectAndCompute(g, None)
    t_s_full = (time.perf_counter() - t0) * 1000.0
    
    # SIFT 1000
    t0 = time.perf_counter()
    ks_1k, ds_1k = sift_1000.detectAndCompute(g, None)
    t_s_1k = (time.perf_counter() - t0) * 1000.0
    
    # ORB 1000
    t0 = time.perf_counter()
    ko_1k, do_1k = orb_1000.detectAndCompute(g, None)
    t_o_1k = (time.perf_counter() - t0) * 1000.0
    
    comparison_rows.append([
        fname,
        len(ks_full),
        f"{t_s_full:.1f}",
        len(ks_1k),
        f"{ds_1k.shape[1] if ds_1k is not None else 0}",
        f"{ds_1k.dtype if ds_1k is not None else '-'}",
        f"{t_s_1k:.1f}",
        len(ko_1k),
        f"{do_1k.shape[1] if do_1k is not None else 0}",
        f"{do_1k.dtype if do_1k is not None else '-'}",
        f"{t_o_1k:.1f}",
        f"{(t_s_1k / t_o_1k):.1f}x"
    ])

save_csv(
    os.path.join(RESULTS_DIR, "c_detector_benchmarks.csv"),
    [
        "Image", "SIFT_Full_Count", "SIFT_Full_Time_ms",
        "SIFT_1000_Count", "SIFT_Desc_Dim", "SIFT_Desc_Dtype", "SIFT_1000_Time_ms",
        "ORB_1000_Count", "ORB_Desc_Dim", "ORB_Desc_Dtype", "ORB_1000_Time_ms",
        "ORB_Speedup"
    ],
    comparison_rows
)
print("Збережено c_detector_benchmarks.csv")

# =========================================================================
# Крок 13: Візуалізація з прапорцем DRAW_RICH_KEYPOINTS
# =========================================================================
# Для найкращої наочності обираємо фрагмент розміром 800x600 пікселів у зоні картин
cy, cx = h // 2, w // 2
crop_y1, crop_y2 = cy - 200, cy + 300
crop_x1, crop_x2 = cx - 350, cx + 350

# Генеруємо зображення з rich keypoints на всьому кадрі
img_sift_rich = cv2.drawKeypoints(
    base_bgr, kps_sift1k, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    color=(0, 220, 50)
)
img_orb_rich = cv2.drawKeypoints(
    base_bgr, kps_orb1k, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    color=(240, 50, 160)
)

# Переводимо в RGB для matplotlib
sift_rich_rgb = cv2.cvtColor(img_sift_rich, cv2.COLOR_BGR2RGB)
orb_rich_rgb = cv2.cvtColor(img_orb_rich, cv2.COLOR_BGR2RGB)

# 1. Загальний порівняльний рисунок огляду (2 рядки: зверху SIFT, знизу ORB)
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

axes[0].imshow(sift_rich_rgb)
axes[0].set_title(f"(а) SIFT: Rich Keypoints ($N={len(kps_sift1k)}$, розмір кола = масштаб $\\sigma$, промінь = орієнтація $\\theta$)", fontsize=11)
axes[0].axis("off")

axes[1].imshow(orb_rich_rgb)
axes[1].set_title(f"(б) ORB: Rich Keypoints ($N={len(kps_orb1k)}$, піраміда FAST + кут орієнтації інтенсивності)", fontsize=11)
axes[1].axis("off")

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "c_rich_keypoints_full.png"))
plt.close(fig)
print("Збережено c_rich_keypoints_full.png")

# 2. Детальний кроп-рисунок (великий масштаб для детального розгляду окремих точок)
crop_sift = sift_rich_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
crop_orb = orb_rich_rgb[crop_y1:crop_y2, crop_x1:crop_x2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

ax1.imshow(crop_sift)
ax1.set_title("SIFT (фрагмент з масштабами та орієнтаціями)", fontsize=10)
ax1.axis("off")

ax2.imshow(crop_orb)
ax2.set_title("ORB (фрагмент з FAST-кутами та орієнтаціями)", fontsize=10)
ax2.axis("off")

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "c_rich_keypoints_crop.png"))
plt.close(fig)
print("Збережено c_rich_keypoints_crop.png")

# 3. Розподіл масштабу та орієнтації точок SIFT та ORB
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sift_sizes = [k.size for k in kps_sift1k]
orb_sizes = [k.size for k in kps_orb1k]

axes[0].hist(sift_sizes, bins=30, alpha=0.6, color="#1b9e77", label=f"SIFT (сер. {np.mean(sift_sizes):.1f} пк)", density=True)
axes[0].hist(orb_sizes, bins=30, alpha=0.6, color="#7570b3", label=f"ORB (сер. {np.mean(orb_sizes):.1f} пк)", density=True)
axes[0].set_xlabel("Характерний масштаб (діаметр кола, пк)", fontsize=9)
axes[0].set_ylabel("Густина ймовірності", fontsize=9)
axes[0].set_title("Розподіл просторових масштабів keypoints", fontsize=10)
axes[0].legend(framealpha=0.9)
axes[0].grid(True, linestyle=":", alpha=0.6)

sift_angles = [k.angle for k in kps_sift1k if k.angle >= 0]
orb_angles = [k.angle for k in kps_orb1k if k.angle >= 0]

axes[1].hist(sift_angles, bins=36, range=(0, 360), alpha=0.6, color="#1b9e77", label="SIFT", density=True)
axes[1].hist(orb_angles, bins=36, range=(0, 360), alpha=0.6, color="#7570b3", label="ORB", density=True)
axes[1].set_xlabel("Орієнтація ключової точки, градуси", fontsize=9)
axes[1].set_ylabel("Густина ймовірності", fontsize=9)
axes[1].set_title("Розподіл орієнтацій $\\theta \\in [0^\\circ, 360^\\circ)$", fontsize=10)
axes[1].legend(framealpha=0.9)
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "c_scale_angle_distribution.png"))
plt.close(fig)
print("Збережено c_scale_angle_distribution.png")

# Журнал виконання Частини C
log_path = os.path.join(RESULTS_DIR, "c_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ C: SIFT ТА ORB ===\n")
    f.write(f"Базовий файл: paintings.jpg ({w} x {h})\n")
    f.write("Параметри дескрипторів:\n")
    f.write(f"  SIFT: розмірність = {des_sift1k.shape[1]} (128 floats = 512 байт/точка), тип = {des_sift1k.dtype}\n")
    f.write(f"  ORB:  розмірність = {des_orb1k.shape[1]} (32 uint8 = 256 біт = 32 байти/точка), тип = {des_orb1k.dtype}\n")
    f.write(f"  Обсяг пам'яті на точку: ORB у 16 разів компактніший за SIFT (32 Б проти 512 Б).\n\n")
    f.write("Швидкодія (detectAndCompute на 1000 точок):\n")
    f.write(f"  SIFT (1000): {t_s_1k:.1f} мс\n")
    f.write(f"  ORB (1000):  {t_o_1k:.1f} мс\n")
    f.write(f"  Прискорення ORB: {t_s_1k / t_o_1k:.1f}x\n")

print(f"Збережено {log_path}")
print("Частина C успішно виконана!\n")

