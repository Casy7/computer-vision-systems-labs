"""Лабораторна робота №2.
Частина D: Дослідження збіжності, усунення дисторсії та візуальна перевірка (Кроки 12-15).
"""
import os
import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import (
    ROOT, LAB, DATA, RESULTS, FIGURES, PATTERN_SIZE, SQUARE_SIZE, SUBPIX_CRITERIA,
    imread, rgb, create_object_points, compute_reprojection_errors, save_csv
)

print("=== Частина D: Дослідження збіжності, виправлення дисторсії та експорт ===")

# Завантаження 24 валідних (відфільтрованих) кадрів без викиду angle_8
file_list = []
for grp in ["frontal", "angles", "corners"]:
    flist = sorted(glob.glob(os.path.join(DATA, grp, "*.jpg")))
    for f in flist:
        if "angle_8.jpg" not in f:
            file_list.append((grp, f))

objp = create_object_points(PATTERN_SIZE, SQUARE_SIZE)
clean_objpoints = []
clean_imgpoints = []
h, w = None, None

for grp, fpath in file_list:
    img = imread(fpath)
    if h is None:
        h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)
    corners_sp = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    clean_objpoints.append(objp)
    clean_imgpoints.append(corners_sp.reshape(-1, 2))

# Базове калібрування на всіх 24 чистих кадрах
ret_final, K_final, D_final, rvecs_final, tvecs_final = cv2.calibrateCamera(
    clean_objpoints, clean_imgpoints, (w, h), None, None
)
print(f"Підсумкове калібрування (24 кадри): RMSE = {ret_final:.4f} px")

# =========================================================================
# Крок 12: Залежність похибки та fx від кількості кадрів N
# =========================================================================
np.random.seed(42)
random.seed(42)

Ns = [5, 8, 10, 12, 15, 20]
trials = 5
dependence_rows = []

n_stats = {}

for N in Ns:
    err_list = []
    fx_list = []
    fy_list = []
    for t in range(trials):
        idx = np.random.choice(len(clean_objpoints), N, replace=False)
        sub_obj = [clean_objpoints[i] for i in idx]
        sub_img = [clean_imgpoints[i] for i in idx]
        ret, K, D, _, _ = cv2.calibrateCamera(sub_obj, sub_img, (w, h), None, None)
        err_list.append(ret)
        fx_list.append(K[0, 0])
        fy_list.append(K[1, 1])

    mean_err, std_err = np.mean(err_list), np.std(err_list)
    mean_fx, std_fx = np.mean(fx_list), np.std(fx_list)
    mean_fy, std_fy = np.mean(fy_list), np.std(fy_list)

    n_stats[N] = {
        "err_mean": mean_err, "err_std": std_err,
        "fx_mean": mean_fx, "fx_std": std_fx,
        "fy_mean": mean_fy, "fy_std": std_fy
    }
    dependence_rows.append([
        N, f"{mean_err:.4f}", f"{std_err:.4f}",
        f"{mean_fx:.2f}", f"{std_fx:.2f}",
        f"{mean_fy:.2f}", f"{std_fy:.2f}"
    ])
    print(f"N = {N:2d}: RMSE = {mean_err:.4f} +/- {std_err:.4f} px, fx = {mean_fx:.2f} +/- {std_fx:.2f} px")

save_csv(
    os.path.join(RESULTS, "d_n_frames_dependence.csv"),
    ["N", "RMSE_mean", "RMSE_std", "fx_mean", "fx_std", "fy_mean", "fy_std"],
    dependence_rows
)
print("Збережено d_n_frames_dependence.csv")

# Побудова графіка виходу на плато (Крок 12)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Графік 1: fx від N
ax1.errorbar(
    Ns, [n_stats[N]["fx_mean"] for N in Ns],
    yerr=[n_stats[N]["fx_std"] for N in Ns],
    fmt="o-", color="#2b5c8f", ecolor="#2b5c8f", elinewidth=1.5, capsize=4,
    label="Оцінка $f_x$ (середнє $\\pm$ СВ)"
)
ax1.axhline(K_final[0, 0], color="red", linestyle="--", alpha=0.8,
            label=f"Еталон (24 кадри, {K_final[0, 0]:.1f} пк)")
ax1.axvspan(12, 20, color="green", alpha=0.08, label="Зона плато ($N \\geq 12$)")
ax1.set_xlabel("Кількість кадрів калібрування $N$", fontsize=10)
ax1.set_ylabel("Фокусна відстань $f_x$, пікселі", fontsize=10)
ax1.set_title("Збіжність оцінки $f_x$ зі зростанням $N$", fontsize=11)
ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax1.grid(True, linestyle=":", alpha=0.6)

# Графік 2: Похибка репроєкції від N
ax2.errorbar(
    Ns, [n_stats[N]["err_mean"] for N in Ns],
    yerr=[n_stats[N]["err_std"] for N in Ns],
    fmt="s-", color="#d95f02", ecolor="#d95f02", elinewidth=1.5, capsize=4,
    label="Похибка RMSE (середнє $\\pm$ СВ)"
)
ax2.axhline(ret_final, color="red", linestyle="--", alpha=0.8,
            label=f"Еталон ({ret_final:.3f} пк)")
ax2.axvspan(12, 20, color="green", alpha=0.08, label="Зона плато ($N \\geq 12$)")
ax2.set_xlabel("Кількість кадрів калібрування $N$", fontsize=10)
ax2.set_ylabel("Похибка репроєкції RMSE, пікселі", fontsize=10)
ax2.set_title("Стабілізація похибки репроєкції від $N$", fontsize=11)
ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES, "d_n_frames_convergence.png"), dpi=160)
plt.close(fig)
print("Збережено d_n_frames_convergence.png")

# =========================================================================
# Крок 13: Усунення дисторсії: cv2.undistort з alpha = 0 та alpha = 1
# =========================================================================
sample_img_path = os.path.join(DATA, "corners", "corner_7.jpg")
sample_img = imread(sample_img_path)

# Матриці для alpha = 0 (crop без чорних полів) та alpha = 1 (збереження всього поля з полями)
newcameramtx_0, roi_0 = cv2.getOptimalNewCameraMatrix(K_final, D_final, (w, h), 0, (w, h))
newcameramtx_1, roi_1 = cv2.getOptimalNewCameraMatrix(K_final, D_final, (w, h), 1, (w, h))

undist_0 = cv2.undistort(sample_img, K_final, D_final, None, newcameramtx_0)
undist_1 = cv2.undistort(sample_img, K_final, D_final, None, newcameramtx_1)

# Візуальне порівняння трьох режимів
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

axes[0].imshow(rgb(cv2.resize(sample_img, (800, 600))))
axes[0].set_title("Вихідне зображення (з дисторсією)", fontsize=9)

axes[1].imshow(rgb(cv2.resize(undist_0, (800, 600))))
axes[1].set_title("Виправлене: $\\alpha = 0$ (кадр обрізано)", fontsize=9)

axes[2].imshow(rgb(cv2.resize(undist_1, (800, 600))))
axes[2].set_title("Виправлене: $\\alpha = 1$ (з чорними полями)", fontsize=9)

for ax in axes:
    ax.axis("off")

plt.tight_layout()
fig.savefig(os.path.join(FIGURES, "d_undistort_alpha.png"), dpi=160)
plt.close(fig)
print("Збережено d_undistort_alpha.png")

# =========================================================================
# Крок 14: Візуальна перевірка прямолінійності на периферійній лінії
# =========================================================================
# На кадрі angle_10 рядок кутів 5 проходить через периферійну зону сильної дисторсії
line_img_path = os.path.join(DATA, "angles", "angle_10.jpg")
line_img = imread(line_img_path)
gray_line = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
ret_chk, corners_chk = cv2.findChessboardCorners(gray_line, PATTERN_SIZE)
corners_chk_sp = cv2.cornerSubPix(gray_line, corners_chk, (11, 11), (-1, -1), SUBPIX_CRITERIA)

# Вибираємо рядок кутів поблизу краю (рядок 5 з 9 точок)
pts_orig = corners_chk_sp.reshape(6, 9, 2)[5] # 9 точок уздовж прямолінійного краю шаблона

# Координати точок після усунення дисторсії (P=K зберігає ту саму матрицю)
pts_undist = cv2.undistortPoints(pts_orig.reshape(-1, 1, 2), K_final, D_final, P=K_final).reshape(9, 2)

# Функція розрахунку відхилення (стріли прогину / сагіти) від ідеальної прямої між крайніми точками
def compute_deflections(points):
    p_start, p_end = points[0], points[-1]
    vec = p_end - p_start
    length = np.hypot(vec[0], vec[1])
    normal = np.array([-vec[1], vec[0]]) / length
    deflections = [float(np.dot(points[i] - p_start, normal)) for i in range(len(points))]
    return deflections

deflections_orig = compute_deflections(pts_orig)
deflections_undist = compute_deflections(pts_undist)

max_dev_orig = np.max(np.abs(deflections_orig))
max_dev_undist = np.max(np.abs(deflections_undist))
print(f"Крок 14: Максимальне викривлення лінії ДО виправлення: {max_dev_orig:.2f} px")
print(f"Крок 14: Максимальне викривлення лінії ПІСЛЯ виправлення: {max_dev_undist:.2f} px")

# Побудова порівняльного рисунка: фрагмент кадру з накладеною прямою та графік відхилень
fig, (ax_img1, ax_img2, ax_plot) = plt.subplots(1, 3, figsize=(14, 4.2))

# Малювання прямої опорної хорди на оригінальному фрагменті
img_line_orig = line_img.copy()
cv2.line(
    img_line_orig,
    tuple(pts_orig[0].astype(int)),
    tuple(pts_orig[-1].astype(int)),
    (0, 0, 255), 4, cv2.LINE_AA
)
for pt in pts_orig:
    cv2.circle(img_line_orig, tuple(pt.astype(int)), 8, (0, 255, 0), -1)

# На виправленому зображенні
img_undist_full = cv2.undistort(line_img, K_final, D_final)
cv2.line(
    img_undist_full,
    tuple(pts_undist[0].astype(int)),
    tuple(pts_undist[-1].astype(int)),
    (0, 0, 255), 4, cv2.LINE_AA
)
for pt in pts_undist:
    cv2.circle(img_undist_full, tuple(pt.astype(int)), 8, (0, 255, 0), -1)

# Збільшений фрагмент області лінії
x_min = int(min(pts_orig[:, 0].min(), pts_undist[:, 0].min()) - 100)
x_max = int(max(pts_orig[:, 0].max(), pts_undist[:, 0].max()) + 100)
y_min = int(min(pts_orig[:, 1].min(), pts_undist[:, 1].min()) - 120)
y_max = int(max(pts_orig[:, 1].max(), pts_undist[:, 1].max()) + 120)

x_min, x_max = max(0, x_min), min(w, x_max)
y_min, y_max = max(0, y_min), min(h, y_max)

crop_orig = img_line_orig[y_min:y_max, x_min:x_max]
crop_undist = img_undist_full[y_min:y_max, x_min:x_max]

ax_img1.imshow(rgb(crop_orig))
ax_img1.set_title(f"ДО виправлення (викривлення {max_dev_orig:.2f} пк)", fontsize=9)
ax_img1.axis("off")

ax_img2.imshow(rgb(crop_undist))
ax_img2.set_title(f"ПІСЛЯ виправлення (викривлення {max_dev_undist:.2f} пк)", fontsize=9)
ax_img2.axis("off")

# Графік відхилень від прямої уздовж 9 точок
node_indices = np.arange(1, 10)
ax_plot.plot(node_indices, deflections_orig, "ro--", label=f"До виправлення (макс {max_dev_orig:.2f} пк)")
ax_plot.plot(node_indices, deflections_undist, "bs-", label=f"Після виправлення (макс {max_dev_undist:.2f} пк)")
ax_plot.axhline(0, color="black", linestyle=":", linewidth=0.8)
ax_plot.set_xlabel("Номер вузла вздовж лінії", fontsize=9)
ax_plot.set_ylabel("Відхилення від ідеальної прямої, пк", fontsize=9)
ax_plot.set_title("Профіль прямолінійності хорди", fontsize=10)
ax_plot.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax_plot.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES, "d_straight_line_check.png"), dpi=160)
plt.close(fig)
print("Збережено d_straight_line_check.png")

# Збереження даних перевірки прямолінійності
straight_rows = [
    [i + 1, f"{deflections_orig[i]:.3f}", f"{deflections_undist[i]:.3f}"]
    for i in range(9)
]
save_csv(
    os.path.join(RESULTS, "d_straight_line_profile.csv"),
    ["Вузол", "Відхилення_до_пк", "Відхилення_після_пк"],
    straight_rows
)
print("Збережено d_straight_line_profile.csv")

# =========================================================================
# Крок 15: Збереження camera_params.yml через cv2.FileStorage
# =========================================================================
yaml_path = os.path.join(RESULTS, "camera_params.yml")
fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)

fs.write("calibration_date", "2026-10-29")
fs.write("camera_model", "Google Pixel 5")
fs.write("image_width", int(w))
fs.write("image_height", int(h))
fs.write("num_calibration_frames", int(len(clean_objpoints)))
fs.write("reprojection_error_rmse", float(ret_final))
fs.write("square_size_mm", float(SQUARE_SIZE))
fs.write("pattern_size_cols", int(PATTERN_SIZE[0]))
fs.write("pattern_size_rows", int(PATTERN_SIZE[1]))
fs.write("camera_matrix", K_final)
fs.write("distortion_coefficients", D_final)

fs.release()
print(f"Збережено {yaml_path}")

# Також дублюємо у кореневий каталог lab02/camera_params.yml для легкого доступу
root_yml = os.path.join(LAB, "camera_params.yml")
fs2 = cv2.FileStorage(root_yml, cv2.FILE_STORAGE_WRITE)
fs2.write("calibration_date", "2026-10-29")
fs2.write("camera_model", "Google Pixel 5")
fs2.write("image_width", int(w))
fs2.write("image_height", int(h))
fs2.write("num_calibration_frames", int(len(clean_objpoints)))
fs2.write("reprojection_error_rmse", float(ret_final))
fs2.write("square_size_mm", float(SQUARE_SIZE))
fs2.write("camera_matrix", K_final)
fs2.write("distortion_coefficients", D_final)
fs2.release()
print(f"Збережено {root_yml}")

# Запис підсумкового текстового журналу
log_path = os.path.join(RESULTS, "calibration_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== ЗВІТ ПРО КАЛІБРУВАННЯ КАМЕРИ (ЛАБОРАТОРНА РОБОТА №2) ===\n")
    f.write(f"Дата виконання: 2026-10-29\n")
    f.write(f"Камера: Google Pixel 5\n")
    f.write(f"Фізична роздільна здатність: {w} x {h}\n")
    f.write(f"Шаблон: {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} внутрішніх кутів, клітина = {SQUARE_SIZE} мм\n")
    f.write(f"Всього кадрів у наборі: 25\n")
    f.write(f"Вилучено викидів: 1 (angle_8.jpg, похибка 2.951 пк > 2x медіани 2.767 пк)\n")
    f.write(f"Використано для фінальної моделі: 24 кадри\n")
    f.write(f"Підсумкова похибка репроєкції (RMSE): {ret_final:.4f} пікселів\n\n")
    f.write(f"Матриця камери K:\n{K_final}\n\n")
    f.write(f"Коефіцієнти дисторсії D:\n{D_final.ravel()}\n\n")
    f.write(f"fx = {K_final[0,0]:.2f}, fy = {K_final[1,1]:.2f}, fx/fy = {K_final[0,0]/K_final[1,1]:.4f}\n")
    f.write(f"cx = {K_final[0,2]:.2f} (відхилення від центру {abs(K_final[0,2]-w/2)/(w/2)*100:.2f}%)\n")
    f.write(f"cy = {K_final[1,2]:.2f} (відхилення від центру {abs(K_final[1,2]-h/2)/(h/2)*100:.2f}%)\n")

print(f"Збережено {log_path}")
print("Частина D успішно виконана!\n")
