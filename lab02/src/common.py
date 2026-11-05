"""Спільні утиліти та константи для Лабораторної роботи №2."""
import os
import sys
import numpy as np
import cv2
import matplotlib

sys.stdout.reconfigure(encoding="utf-8")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAB = os.path.join(ROOT, "lab02")
DATA = os.path.join(LAB, "data")
RESULTS = os.path.join(LAB, "results")
FIGURES = os.path.join(LAB, "figures")

for d in (RESULTS, FIGURES):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight"
})

# Параметри калібрувального шаблона згідно з Практикумом та умовами
PATTERN_SIZE = (9, 6)   # 9 внутрішніх кутів по ширині, 6 по висоті (10x7 клітин)
SQUARE_SIZE = 22.0      # реальний виміряний розмір сторони клітини: 2.2 см = 22.0 мм

# Критерій зупинки для cv2.cornerSubPix: 30 ітерацій або точність 0.001
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def imread(path, flags=cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION):
    """Зчитування зображення в системі координат матриці сенсора (без довільних поворотів EXIF)."""
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Не вдалося зчитати файл: {path}")
    return img


def rgb(img):
    """BGR -> RGB для коректного виведення в matplotlib."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_object_points(pattern_size=PATTERN_SIZE, square_size=SQUARE_SIZE):
    """Формування тривимірних координат кутів шахівниці (Z = 0) у міліметрах."""
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    return objp


def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, D):
    """Ручний розрахунок похибки репроєкції для кожного кадру та сумарної RMSE."""
    per_frame_mean = []
    per_frame_rmse = []
    all_sq_errors = []

    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(-1, 2)
        pts = imgpoints[i].reshape(-1, 2)
        
        diff = pts - proj
        sq_err = np.sum(diff ** 2, axis=1)
        all_sq_errors.extend(sq_err)
        
        dists = np.sqrt(sq_err)
        per_frame_mean.append(float(np.mean(dists)))
        per_frame_rmse.append(float(np.sqrt(np.mean(sq_err))))

    total_rmse = float(np.sqrt(np.mean(all_sq_errors)))
    return per_frame_mean, per_frame_rmse, total_rmse


def save_csv(path, header, rows):
    """Збереження даних у CSV файл з кодуванням UTF-8."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

