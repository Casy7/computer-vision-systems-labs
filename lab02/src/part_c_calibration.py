"""Лабораторна робота №2.
Частина C: Калібрування камери, субпіксельне уточнення, похибка репроєкції та фільтрація викидів (Кроки 6-11).
"""
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import (
    DATA, RESULTS, FIGURES, PATTERN_SIZE, SQUARE_SIZE, SUBPIX_CRITERIA,
    imread, rgb, create_object_points, compute_reprojection_errors, save_csv
)

print("=== Частина C: Калібрування камери та аналіз похибки репроєкції ===")

# Список усіх файлів
file_list = []
for grp in ["frontal", "angles", "corners"]:
    flist = sorted(glob.glob(os.path.join(DATA, grp, "*.jpg")))
    for f in flist:
        file_list.append((grp, f))

objp = create_object_points(PATTERN_SIZE, SQUARE_SIZE)

objpoints = []
imgpoints_raw = []
imgpoints_subpix = []
valid_metadata = []
displacement_list = []

sample_corner_vis = None

for grp, fpath in file_list:
    fname = os.path.basename(fpath)
    img = imread(fpath)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Крок 6: Знаходження кутів шахівниці
    ret, corners = cv2.findChessboardCorners(
        gray, PATTERN_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)
        
    if not ret:
        print(f"ПОПЕРЕДЖЕННЯ: Кути не знайдено на кадрі {fname}")
        continue

    # Крок 7: Субпіксельне уточнення
    raw_pts = corners.reshape(-1, 2)
    corners_sp = cv2.cornerSubPix(gray, corners.copy(), (11, 11), (-1, -1), SUBPIX_CRITERIA)
    sp_pts = corners_sp.reshape(-1, 2)

    # Вимірювання зсуву субпіксельного уточнення
    disp = np.linalg.norm(sp_pts - raw_pts, axis=1)
    displacement_list.extend(disp)

    objpoints.append(objp)
    imgpoints_raw.append(raw_pts)
    imgpoints_subpix.append(sp_pts)
    valid_metadata.append((grp, fname, fpath))

    # Зберігаємо візуалізацію для одного з репрезентативних кадрів
    if fname == "scale_1.jpg":
        vis_img = img.copy()
        cv2.drawChessboardCorners(vis_img, PATTERN_SIZE, corners_sp, True)
        sample_corner_vis = vis_img

print(f"Успішно детектовано кути на {len(valid_metadata)} / {len(file_list)} кадрах.")

# Крок 7 & 9: Калібрування без субпіксельного уточнення
ret_raw, K_raw, D_raw, rvecs_raw, tvecs_raw = cv2.calibrateCamera(
    objpoints, imgpoints_raw, (w, h), None, None
)
_, _, manual_rmse_raw = compute_reprojection_errors(
    objpoints, imgpoints_raw, rvecs_raw, tvecs_raw, K_raw, D_raw
)

# Крок 7 & 9: Калібрування з субпіксельним уточненням (усі 25 кадрів)
ret_sp, K_sp, D_sp, rvecs_sp, tvecs_sp = cv2.calibrateCamera(
    objpoints, imgpoints_subpix, (w, h), None, None
)
per_frame_mean, per_frame_rmse, manual_rmse_sp = compute_reprojection_errors(
    objpoints, imgpoints_subpix, rvecs_sp, tvecs_sp, K_sp, D_sp
)

print(f"Калібрування (без субпікселя): RMSE = {ret_raw:.4f} px (ручний = {manual_rmse_raw:.4f} px)")
print(f"Калібрування (з субпікселем):   RMSE = {ret_sp:.4f} px (ручний = {manual_rmse_sp:.4f} px)")

# Крок 10: Похибка по кадрах та визначення викидів
med_err = float(np.median(per_frame_mean))
threshold = 2.0 * med_err
print(f"Медіанна похибка: {med_err:.4f} px, Поріг відсікання (2x): {threshold:.4f} px")

frame_errors_data = []
outlier_indices = []

for idx, (grp, fname, fpath) in enumerate(valid_metadata):
    m_err = per_frame_mean[idx]
    r_err = per_frame_rmse[idx]
    is_outlier = m_err > threshold
    if is_outlier:
        outlier_indices.append(idx)
    frame_errors_data.append([
        grp, fname, f"{m_err:.4f}", f"{r_err:.4f}", "OUTLIER" if is_outlier else "OK"
    ])

save_csv(
    os.path.join(RESULTS, "c_per_frame_errors.csv"),
    ["Група", "Файл", "Середня_похибка_px", "RMSE_px", "Статус"],
    frame_errors_data
)
print(f"Збережено c_per_frame_errors.csv. Знайдено викидів: {len(outlier_indices)}")
for idx in outlier_indices:
    print(f"  -> Викид: {valid_metadata[idx][1]} (похибка {per_frame_mean[idx]:.4f} px)")

# Крок 11: Вилучення викидів і перерахунок калібрування
clean_objpoints = [objpoints[i] for i in range(len(objpoints)) if i not in outlier_indices]
clean_imgpoints = [imgpoints_subpix[i] for i in range(len(imgpoints_subpix)) if i not in outlier_indices]
clean_metadata = [valid_metadata[i] for i in range(len(valid_metadata)) if i not in outlier_indices]

ret_clean, K_clean, D_clean, rvecs_clean, tvecs_clean = cv2.calibrateCamera(
    clean_objpoints, clean_imgpoints, (w, h), None, None
)
clean_mean, clean_rmse, manual_rmse_clean = compute_reprojection_errors(
    clean_objpoints, clean_imgpoints, rvecs_clean, tvecs_clean, K_clean, D_clean
)

print(f"Калібрування після фільтрації ({len(clean_objpoints)} кадрів): RMSE = {ret_clean:.4f} px")
print(f"Матриця камери K:\n{K_clean}")
print(f"Коефіцієнти дисторсії D: {D_clean.ravel()}")

# Порівняльна таблиця результатів трьох етапів
def format_calib_row(label, ret, K, D, n_frames):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    d = D.ravel()
    k1, k2, p1, p2, k3 = d[0], d[1], d[2], d[3], d[4]
    return [
        label, n_frames, f"{ret:.4f}", f"{fx:.2f}", f"{fy:.2f}", f"{fx/fy:.4f}",
        f"{cx:.2f}", f"{cy:.2f}", f"{k1:.4f}", f"{k2:.4f}", f"{p1:.5f}", f"{p2:.5f}", f"{k3:.4f}"
    ]

comparison_rows = [
    format_calib_row("Без субпікселя", ret_raw, K_raw, D_raw, len(objpoints)),
    format_calib_row("З субпікселем (всі)", ret_sp, K_sp, D_sp, len(objpoints)),
    format_calib_row("Після фільтрації (чисті)", ret_clean, K_clean, D_clean, len(clean_objpoints))
]
save_csv(
    os.path.join(RESULTS, "c_calibration_comparison.csv"),
    ["Етап", "Кадрів", "RMSE_px", "fx", "fy", "fx_fy_ratio", "cx", "cy", "k1", "k2", "p1", "p2", "k3"],
    comparison_rows
)
print("Збережено c_calibration_comparison.csv")

# Побудова графіка похибок по кадрах (Крок 10)
fig, ax = plt.subplots(figsize=(10, 4.5))
x_indices = np.arange(len(valid_metadata))
bar_colors = ["#d95f02" if i in outlier_indices else "#2b5c8f" for i in range(len(valid_metadata))]

bars = ax.bar(x_indices, per_frame_mean, color=bar_colors, width=0.65, edgecolor="black", linewidth=0.5)

# Лінія порогу 2x медіани та лінія медіани
ax.axhline(med_err, color="green", linestyle="--", linewidth=1.2, label=f"Медіана ({med_err:.2f} пк)")
ax.axhline(threshold, color="red", linestyle=":", linewidth=1.5, label=f"Поріг 2×медіани ({threshold:.2f} пк)")

ax.set_xticks(x_indices)
ax.set_xticklabels([m[1].replace(".jpg", "") for m in valid_metadata], rotation=65, ha="right", fontsize=8)
ax.set_xlabel("Кадри набору даних", fontsize=10)
ax.set_ylabel("Середня похибка репроєкції, пікселі", fontsize=10)
ax.set_title("Похибка репроєкції за окремими кадрами та виявлення викидів", fontsize=11, pad=10)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.6)

# Підпис величини викиду
for idx in outlier_indices:
    ax.annotate(
        f"Викид!\n{per_frame_mean[idx]:.2f} пк",
        xy=(idx, per_frame_mean[idx]),
        xytext=(idx - 1.5, per_frame_mean[idx] + 0.3),
        arrowprops=dict(facecolor="red", arrowstyle="->", lw=1.2),
        fontsize=8, fontweight="bold", color="darkred"
    )

ax.set_ylim(0, max(per_frame_mean) * 1.25)
fig.savefig(os.path.join(FIGURES, "c_per_frame_errors.png"), dpi=160)
plt.close(fig)
print("Збережено c_per_frame_errors.png")

# Побудова графіка розподілу субпіксельного зсуву
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(displacement_list, bins=35, color="#386cb0", edgecolor="black", alpha=0.75, density=True)
ax.axvline(np.mean(displacement_list), color="red", linestyle="--",
           label=f"Середній зсув ({np.mean(displacement_list):.3f} пк)")
ax.set_xlabel("Величина субпіксельного зміщення, пікселі", fontsize=10)
ax.set_ylabel("Густина ймовірності", fontsize=10)
ax.set_title("Розподіл поправок cv2.cornerSubPix ($N = 1350$ точок)", fontsize=11)
ax.legend(framealpha=0.9)
ax.grid(True, linestyle=":", alpha=0.6)
fig.savefig(os.path.join(FIGURES, "c_subpix_displacement.png"), dpi=160)
plt.close(fig)
print("Збережено c_subpix_displacement.png")

# Візуалізація детектування кутів на sample_corner_vis
if sample_corner_vis is not None:
    # Зменшуємо для звіту
    crop_h, crop_w = sample_corner_vis.shape[:2]
    vis_small = cv2.resize(sample_corner_vis, (1200, int(1200 * crop_h / crop_w)))
    cv2.imwrite(os.path.join(FIGURES, "c_corners_detected.png"), vis_small)
    print("Збережено c_corners_detected.png")

print("Частина C успішно завершена!\n")

