"""
Спільний модуль налаштувань, метрик та утиліт для Лабораторної роботи № 3.
«Детектори та дескриптори локальних ознак».
"""
import os
import sys
import numpy as np
import cv2
import matplotlib
from scipy.spatial import cKDTree

# Налаштування кодування та matplotlib
sys.stdout.reconfigure(encoding="utf-8")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Фіксація випадкового зерна для повної відтворюваності
np.random.seed(42)

# Шляхи проекту
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(LAB_DIR, ".."))

DATA_DIR = os.path.join(LAB_DIR, "data")
RESULTS_DIR = os.path.join(LAB_DIR, "results")
FIGURES_DIR = os.path.join(LAB_DIR, "figures")
REPORT_DIR = os.path.join(LAB_DIR, "report")

for d in (RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

# Робоча роздільність для стабільного порівняння детекторів у Частині D
WORKING_MAX_DIM = 1200

# Стилі для наукових графіків за ДСТУ
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "savefig.bbox": "tight"
})

# Кольорова палітра для уніфікованих графіків трьох детекторів
DETECTOR_COLORS = {
    "Harris": "#d95f02",  # червоно-помаранчевий
    "SIFT":   "#1b9e77",  # зелений
    "ORB":    "#7570b3"   # фіолетовий
}

DETECTOR_MARKERS = {
    "Harris": "o",
    "SIFT":   "s",
    "ORB":    "^"
}

DETECTOR_LINESTYLES = {
    "Harris": "-",
    "SIFT":   "--",
    "ORB":    "-."
}


def imread_bgr(path):
    """Зчитування зображення в BGR без неконтрольованих поворотів EXIF."""
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"Не вдалося зчитати файл: {path}")
    return img


def imread_rgb(path):
    """Зчитування зображення в RGB."""
    bgr = imread_bgr(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imread_gray(path):
    """Зчитування зображення у градаціях сірого."""
    bgr = imread_bgr(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# =========================================================================
# Реалізація детектора Гарріса за кроками Практикуму
# =========================================================================

def compute_harris_response(gray, ksize=3, sigma=1.5, k=0.04):
    """
    Обчислення матриці відгуку Гарріса за алгоритмом практикуму (Кроки 1-5):
    1. Переведення у float32
    2. Похідні Ix, Iy через cv2.Sobel(ksize=3)
    3. Поелементні добутки Ix^2, Iy^2, Ix*Iy (до згладжування!)
    4. Гауссове згладжування з sigma=1.5 (cv2.GaussianBlur)
    5. Відгук R = (A*B - C^2) - k*(A+B)^2
    """
    if gray.dtype != np.float32:
        gray_f = gray.astype(np.float32)
    else:
        gray_f = gray

    # Крок 2: оператор Собеля
    Ix = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=ksize)

    # Крок 3: поелементні квадрати та добуток
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Крок 4: гауссове згладжування
    A = cv2.GaussianBlur(Ixx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    B = cv2.GaussianBlur(Iyy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    C = cv2.GaussianBlur(Ixy, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Крок 5: відгук Гарріса
    detM = A * B - C * C
    traceM = A + B
    R = detM - k * (traceM ** 2)

    return R, (Ix, Iy), (A, B, C)


def nms_corners(R, max_points=1000, radius=10, rel_thresh=0.01):
    """
    Ізотропне немаксимальне придушення (NMS) за Кроками 6-7 практикуму
    з використанням евклідової кругової маски (L2-норма замість прямокутної L_inf):
    1. Відсікання точок із R <= rel_thresh * max(R)
    2. Сортування точок за спаданням R
    3. Кругове евклідове придушення точок у радіусі radius.
    4. Зупинка при досягненні max_points точок.
    """
    max_val = float(R.max())
    if max_val <= 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    thresh = rel_thresh * max_val
    ys, xs = np.where(R > thresh)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    vals = R[ys, xs]
    order = np.argsort(-vals)
    ys = ys[order]
    xs = xs[order]
    vals = vals[order]

    h, w = R.shape
    suppressed = np.zeros((h, w), dtype=bool)
    selected_pts = []
    selected_vals = []
    r = int(round(radius))

    # Кругова булева маска (xx^2 + yy^2 <= r^2)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    circle_mask = (xx ** 2 + yy ** 2 <= r ** 2)

    for y, x, val in zip(ys, xs, vals):
        if not suppressed[y, x]:
            selected_pts.append((float(x), float(y)))
            selected_vals.append(float(val))

            # Межі зрізу на зображенні
            y_min = max(0, y - r)
            y_max = min(h, y + r + 1)
            x_min = max(0, x - r)
            x_max = min(w, x + r + 1)

            # Відповідні межі зрізу кругової маски
            cy_min = y_min - (y - r)
            cy_max = cy_min + (y_max - y_min)
            cx_min = x_min - (x - r)
            cx_max = cx_min + (x_max - x_min)

            suppressed[y_min:y_max, x_min:x_max] |= circle_mask[cy_min:cy_max, cx_min:cx_max]

            if max_points is not None and len(selected_pts) >= max_points:
                break

    return np.array(selected_pts, dtype=np.float32), np.array(selected_vals, dtype=np.float32)


def circular_nms_points(pts, responses, max_points=1000, radius=10, img_shape=None):
    """
    Ізотропне кругове NMS для довільного набору точок із response (SIFT, ORB).
    """
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)

    order = np.argsort(-responses)
    pts = pts[order]
    responses = responses[order]

    h, w = img_shape
    suppressed = np.zeros((h, w), dtype=bool)
    selected_pts = []
    r = max(2, int(round(radius)))

    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    circle_mask = (xx ** 2 + yy ** 2 <= r ** 2)

    for pt in pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= y < h and 0 <= x < w and not suppressed[y, x]:
            selected_pts.append(pt)
            y_min = max(0, y - r)
            y_max = min(h, y + r + 1)
            x_min = max(0, x - r)
            x_max = min(w, x + r + 1)

            cy_min = y_min - (y - r)
            cy_max = cy_min + (y_max - y_min)
            cx_min = x_min - (x - r)
            cx_max = cx_min + (x_max - x_min)

            suppressed[y_min:y_max, x_min:x_max] |= circle_mask[cy_min:cy_max, cx_min:cx_max]

            if max_points is not None and len(selected_pts) >= max_points:
                break

    return np.array(selected_pts, dtype=np.float32)


def detect_custom_harris(gray, max_points=1000, radius=10, rel_thresh=0.01, sigma=1.5, k=0.04):
    """Повний конвеєр власного детектора Гарріса."""
    R, _, _ = compute_harris_response(gray, ksize=3, sigma=sigma, k=k)
    pts, responses = nms_corners(R, max_points=max_points, radius=radius, rel_thresh=rel_thresh)
    return pts, responses, R


def detect_cv_harris(gray, max_points=1000, radius=10, blockSize=3, ksize=3, k=0.04, rel_thresh=0.01):
    """Бібліотечний cv2.cornerHarris із тим самим порогом та ізотропним NMS."""
    if gray.dtype != np.float32:
        gray_f = gray.astype(np.float32)
    else:
        gray_f = gray

    R_cv = cv2.cornerHarris(gray_f, blockSize=blockSize, ksize=ksize, k=k)
    pts, responses = nms_corners(R_cv, max_points=max_points, radius=radius, rel_thresh=rel_thresh)
    return pts, responses, R_cv


def detect_sift_keypoints(gray, max_points=1000, radius=10, min_size=5.0):
    """
    Детектор SIFT з відбором фізично значущих точок (фільтрація наддрібного шуму першої октави kp.size < min_size)
    та ізотропним круговим NMS.
    """
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(gray, None)
    if kps is None or len(kps) == 0:
        return np.empty((0, 2), dtype=np.float32), [], None

    # Відсікаємо мікроскопічний субпіксельний пил з октави -1, якщо задано min_size
    valid_indices = [i for i, k in enumerate(kps) if k.size >= min_size]
    if len(valid_indices) < max_points // 2:
        # Якщо фільтр занадто жорсткий для малого зображення, беремо всі
        valid_indices = list(range(len(kps)))

    kps_filt = [kps[i] for i in valid_indices]
    des_filt = des[valid_indices] if des is not None else None

    pts = np.array([k.pt for k in kps_filt], dtype=np.float32)
    resp = np.array([k.response for k in kps_filt], dtype=np.float32)

    selected_pts = circular_nms_points(pts, resp, max_points=max_points, radius=radius, img_shape=gray.shape)

    # Відбираємо відповідні kps та дескриптори
    selected_kps = []
    selected_des = []
    if len(selected_pts) > 0:
        tree = cKDTree(pts)
        _, indices = tree.query(selected_pts)
        for idx in indices:
            selected_kps.append(kps_filt[idx])
            if des_filt is not None:
                selected_des.append(des_filt[idx])

    des_arr = np.array(selected_des) if len(selected_des) > 0 else None
    return selected_pts, selected_kps, des_arr


def detect_orb_keypoints(gray, max_points=1000, radius=10):
    """
    Детектор ORB з відбором найкращих точок за response та ізотропним круговим NMS.
    """
    orb = cv2.ORB_create(nfeatures=max_points * 4)
    kps, des = orb.detectAndCompute(gray, None)
    if kps is None or len(kps) == 0:
        return np.empty((0, 2), dtype=np.float32), [], None

    pts = np.array([k.pt for k in kps], dtype=np.float32)
    resp = np.array([k.response for k in kps], dtype=np.float32)

    selected_pts = circular_nms_points(pts, resp, max_points=max_points, radius=radius, img_shape=gray.shape)

    selected_kps = []
    selected_des = []
    if len(selected_pts) > 0:
        tree = cKDTree(pts)
        _, indices = tree.query(selected_pts)
        for idx in indices:
            selected_kps.append(kps[idx])
            if des is not None:
                selected_des.append(des[idx])

    des_arr = np.array(selected_des) if len(selected_des) > 0 else None
    return selected_pts, selected_kps, des_arr


# =========================================================================
# Відомі геометричні та фотометричні перетворення (Крок 14, 16)
# =========================================================================

def transform_rotation(gray, angle_deg):
    """Поворот зображення на кут angle_deg навколо центру."""
    h, w = gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    H = np.vstack([M, [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(gray, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, H


def transform_scale(gray, scale_factor):
    """
    Масштабування зображення за Кроком 14: діагональна матриця diag(s, s, 1).
    При s < 1.0 використовується cv2.INTER_AREA для усунення аліасингу.
    Розмір полотна дорівнює фізичному розміру масштабованого кадру.
    """
    h, w = gray.shape[:2]
    w_w = max(1, int(round(w * scale_factor)))
    h_w = max(1, int(round(h * scale_factor)))
    H = np.diag([scale_factor, scale_factor, 1.0]).astype(np.float64)

    # При зменшенні - антиаліасинг INTER_AREA, при збільшенні - INTER_LINEAR
    flags = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR
    warped = cv2.warpPerspective(gray, H, (w_w, h_w), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, H


def transform_brightness(gray, factor):
    """Фотометричне перетворення: зміна яскравості множенням на factor."""
    adjusted = np.clip(gray.astype(np.float32) * factor, 0.0, 255.0).astype(gray.dtype)
    H = np.eye(3, dtype=np.float64)
    return adjusted, H


def transform_gaussian_noise(gray, sigma):
    """Додавання адитивного гауссового шуму з параметром sigma."""
    if sigma <= 0.0:
        return gray.copy(), np.eye(3, dtype=np.float64)
    noise = np.random.normal(0.0, sigma, gray.shape).astype(np.float32)
    noisy = np.clip(gray.astype(np.float32) + noise, 0.0, 255.0).astype(gray.dtype)
    H = np.eye(3, dtype=np.float64)
    return noisy, H


# =========================================================================
# Розрахунок метрики повторюваності з One-to-One зіставленням (Крок 15)
# =========================================================================

def compute_repeatability(pts1, pts2, H, shape1, shape2=None, eps=3.0, margin=15):
    """
    Розрахунок повторюваності за формулою практикуму та стандартом Міколайчика-Шмід:
    1. Перенесення точок P1 через матрицю H: p1_w = H * p1
    2. Визначення підмножин точок у межах взаємної видимості (co-visibility) з відступом margin
    3. Жадібне взаємно-однозначне (one-to-one) зіставлення пар за евклідовою відстанню < eps
    4. Нормування на min(|P1_vis|, |P2_vis|)
    """
    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2] if shape2 is not None else (h1, w1)

    if len(pts1) == 0 or len(pts2) == 0:
        return {
            "n1": len(pts1), "n2": len(pts2),
            "n1_vis": 0, "n2_vis": 0,
            "matches": 0, "repeatability": 0.0, "mean_dist": 0.0
        }

    # Пряме перенесення P1 через H
    p1_homo = np.hstack([pts1, np.ones((len(pts1), 1), dtype=np.float64)])
    p1_warped_homo = (H @ p1_homo.T).T
    p1_w = p1_warped_homo[:, :2] / p1_warped_homo[:, 2:3]

    # Зворотне перенесення P2 через H^(-1)
    H_inv = np.linalg.inv(H)
    p2_homo = np.hstack([pts2, np.ones((len(pts2), 1), dtype=np.float64)])
    p2_orig_homo = (H_inv @ p2_homo.T).T
    p2_orig = p2_orig_homo[:, :2] / p2_orig_homo[:, 2:3]

    # Маски точок у межах видимості
    mask1 = (
        (p1_w[:, 0] >= margin) & (p1_w[:, 0] < w2 - margin) &
        (p1_w[:, 1] >= margin) & (p1_w[:, 1] < h2 - margin)
    )
    mask2 = (
        (p2_orig[:, 0] >= margin) & (p2_orig[:, 0] < w1 - margin) &
        (p2_orig[:, 1] >= margin) & (p2_orig[:, 1] < h1 - margin) &
        (pts2[:, 0] >= margin) & (pts2[:, 0] < w2 - margin) &
        (pts2[:, 1] >= margin) & (pts2[:, 1] < h2 - margin)
    )

    pts1_vis = p1_w[mask1]
    pts2_vis = pts2[mask2]

    n1_vis = len(pts1_vis)
    n2_vis = len(pts2_vis)

    if n1_vis == 0 or n2_vis == 0:
        return {
            "n1": len(pts1), "n2": len(pts2),
            "n1_vis": n1_vis, "n2_vis": n2_vis,
            "matches": 0, "repeatability": 0.0, "mean_dist": 0.0
        }

    # Жадібне взаємно-однозначне (One-to-One) зіставлення
    tree2 = cKDTree(pts2_vis)
    dists, idxs = tree2.query(pts1_vis)

    valid_pairs = []
    for i in range(len(dists)):
        if dists[i] < eps:
            valid_pairs.append((dists[i], i, idxs[i]))

    # Сортуємо пари за збільшенням евклідової відстані
    valid_pairs.sort(key=lambda x: x[0])

    matched_p1 = set()
    matched_p2 = set()
    matched_dists = []

    for d, i1, i2 in valid_pairs:
        if i1 not in matched_p1 and i2 not in matched_p2:
            matched_p1.add(i1)
            matched_p2.add(i2)
            matched_dists.append(d)

    matches = len(matched_p1)
    rep = (matches / min(n1_vis, n2_vis)) * 100.0
    mean_dist = float(np.mean(matched_dists)) if matches > 0 else 0.0

    return {
        "n1": len(pts1), "n2": len(pts2),
        "n1_vis": n1_vis, "n2_vis": n2_vis,
        "matches": matches,
        "repeatability": float(rep),
        "mean_dist": mean_dist
    }


def save_csv(path, header, rows):
    """Збереження табличних даних у CSV із кодуванням UTF-8."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
