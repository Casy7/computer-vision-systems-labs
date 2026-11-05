"""
Спільний модуль налаштувань, алгоритмів та утиліт для Лабораторної роботи № 4.
«Зіставлення ознак, RANSAC, побудова панорами».
"""
import os
import sys
import numpy as np
import cv2
import matplotlib
import csv

# Налаштування кодування та бекенду matplotlib
sys.stdout.reconfigure(encoding="utf-8")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Фіксація випадкового зерна для повної відтворюваності експериментів (Крок 5 підготовки)
np.random.seed(42)

# Шляхи проекту
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(LAB_DIR, ".."))

DATA_HW_DIR = os.path.join(ROOT_DIR, "lab01", "data", "Homework")
RESULTS_DIR = os.path.join(LAB_DIR, "results")
FIGURES_DIR = os.path.join(LAB_DIR, "figures")
REPORT_DIR = os.path.join(LAB_DIR, "report")

for d in (RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

# Стилі для наукових графіків за ДСТУ 3008:2015
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 180,
    "savefig.dpi": 250,
    "savefig.bbox": "tight"
})


def imread_bgr(path):
    """Зчитування зображення в колірному просторі BGR без автоповоротів EXIF."""
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"Файл не знайдено або пошкоджено: {path}")
    return img


def imread_rgb(path):
    """Зчитування зображення в RGB."""
    return cv2.cvtColor(imread_bgr(path), cv2.COLOR_BGR2RGB)


def imread_gray(path):
    """Зчитування зображення у градаціях сірого."""
    return cv2.cvtColor(imread_bgr(path), cv2.COLOR_BGR2GRAY)


def save_csv(path, header, rows):
    """Збереження результатів у форматі CSV із кодуванням UTF-8."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


# =========================================================================
# Алгоритми виявлення та зіставлення локальних ознак (Частина A)
# =========================================================================

def extract_sift_features(img_gray, nfeatures=0):
    """
    Обчислення ключових точок та дескрипторів SIFT.
    nfeatures=0 означає збереження всіх знайдених ознак без відсікання.
    """
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = sift.detectAndCompute(img_gray, None)
    return kps, desc


def match_knn(desc1, desc2, k=2):
    """Зіставлення дескрипторів методом k найближчих сусідів (BFMatcher, NORM_L2)."""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=k)
    return matches


def apply_ratio_test(matches_knn, ratio=0.75):
    """Критерій відношення відстаней Лоу (Lowe's ratio test): d1 < ratio * d2."""
    good = []
    for pair in matches_knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def apply_mutual_match(desc1, desc2, ratio=0.75, matches_12=None, matches_21=None):
    """
    Взаємна перевірка зіставлень (Cross-check / Mutual consistency).
    Зіставлення визнається валідним тоді й лише тоді, коли:
    1) desc1[i] є найкращим кандидатом для desc2[j] (з критерієм Лоу);
    2) desc2[j] є найкращим кандидатом для desc1[i] (з критерієм Лоу).
    Для швидкодії matches_12 та matches_21 можуть бути попередньо обчислені.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    if matches_12 is None:
        matches_12 = bf.knnMatch(desc1, desc2, k=2)
    if matches_21 is None:
        matches_21 = bf.knnMatch(desc2, desc1, k=2)

    good_12 = {}
    for pair in matches_12:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_12[m.queryIdx] = m.trainIdx

    good_21 = {}
    for pair in matches_21:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_21[m.queryIdx] = m.trainIdx

    mutual_matches = []
    for pair in matches_12:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                q_idx = m.queryIdx
                t_idx = m.trainIdx
                if good_21.get(t_idx) == q_idx:
                    mutual_matches.append(m)

    return mutual_matches


# =========================================================================
# Геометричні перетворення та оцінка похибки репроєкції
# =========================================================================

def compute_reprojection_errors(src_pts, dst_pts, H):
    """
    Обчислення попіксельної евклідової похибки репроєкції: e_i = ||p2_i - H * p1_i||.
    src_pts, dst_pts: масиви розмірності (N, 2) або (N, 1, 2).
    H: матриця гомографії 3x3.
    """
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)

    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    errors = np.linalg.norm(dst - projected, axis=1)
    return errors


def _frame_corners(shape_hw):
    h, w = shape_hw[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def homography_plausibility(H, shape_src, shape_dst, max_area_ratio=30.0):
    """
    Перевірка геометричної правдоподібності гомографії H: src -> dst.
    Гомографія вважається виродженою, якщо:
      - хоча б один кут кадру src переходить через «лінію горизонту» (знаменник w <= 0);
      - проєкція прямокутника кадру не є опуклим чотирикутником (самоперетин);
      - відношення площ проєкції та кадру поза межами [1/max_area_ratio, max_area_ratio].
    Повертає (ok: bool, reason: str).
    """
    if H is None or not np.all(np.isfinite(H)):
        return False, "H не визначено"
    c = _frame_corners(shape_src)
    hom = np.hstack([c, np.ones((4, 1), np.float32)]) @ H.T
    if np.any(hom[:, 2] <= 1e-9):
        return False, "кут кадру за лінією горизонту"
    proj = (hom[:, :2] / hom[:, 2:3]).astype(np.float32)
    if not cv2.isContourConvex(proj.reshape(-1, 1, 2)):
        return False, "неопукла проєкція кадру"
    area_proj = abs(cv2.contourArea(proj))
    h_s, w_s = shape_src[:2]
    ratio = area_proj / float(w_s * h_s)
    if ratio < 1.0 / max_area_ratio or ratio > max_area_ratio:
        return False, f"зміна площі x{ratio:.3g}"
    return True, "OK"


def compute_overlap_fractions(shape_a, shape_b, H_ab):
    """
    Частки площі перекриття двох кадрів за гомографією H_ab (A -> B):
      cov_b -- яка частка площі кадру B покрита проєкцією кадру A, %;
      cov_a -- яка частка площі кадру A видима в кадрі B, %.
    Для пар зі зміною масштабу ці дві величини принципово різні: збільшений фрагмент
    може бути повністю (100 %) видимим у загальному плані, займаючи там лише кілька відсотків площі.
    Якщо гомографія вироджена, повертає (nan, nan).
    """
    ok, _ = homography_plausibility(H_ab, shape_a, shape_b)
    if not ok:
        return float("nan"), float("nan")
    rect_a, rect_b = _frame_corners(shape_a), _frame_corners(shape_b)
    proj_a = cv2.perspectiveTransform(rect_a.reshape(-1, 1, 2), H_ab).reshape(-1, 2).astype(np.float32)
    inter_b, _ = cv2.intersectConvexConvex(proj_a, rect_b)
    inter_area = max(0.0, float(inter_b))
    cov_b = inter_area / float(shape_b[0] * shape_b[1]) * 100.0
    area_proj_a = max(abs(float(cv2.contourArea(proj_a))), 1e-6)
    cov_a = min(100.0, inter_area / area_proj_a * 100.0)
    return cov_b, cov_a


def homography_corner_distance(H1, H2, shape_hw):
    """Середнє зміщення (пк) чотирьох кутів кадру при перенесенні двома гомографіями H1 та H2."""
    c = _frame_corners(shape_hw).reshape(-1, 1, 2)
    p1 = cv2.perspectiveTransform(c, H1).reshape(-1, 2)
    p2 = cv2.perspectiveTransform(c, H2).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(p1 - p2, axis=1)))


# =========================================================================
# Власний алгоритм RANSAC (Частина C, Кроки 8-12)
# =========================================================================

def _is_valid_quad_sample(p1, p2, min_twice_area=2.0):
    """
    Перевірка невиродженості 4-точкової вибірки для оцінки гомографії:
    жодні 3 точки не колінеарні (площа трикутника > порога), а орієнтація
    чотирьох трикутників зберігається між першим і другим кадром (без самоперетину).
    """
    for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        a1 = (p1[j, 0] - p1[i, 0]) * (p1[k, 1] - p1[i, 1]) - (p1[j, 1] - p1[i, 1]) * (p1[k, 0] - p1[i, 0])
        a2 = (p2[j, 0] - p2[i, 0]) * (p2[k, 1] - p2[i, 1]) - (p2[j, 1] - p2[i, 1]) * (p2[k, 0] - p2[i, 0])
        if abs(a1) < min_twice_area or abs(a2) < min_twice_area or (a1 > 0) != (a2 > 0):
            return False
    return True


def custom_ransac_homography(src_pts, dst_pts, tau=3.0, p=0.99, max_iters=1000, mode="adaptive", seed=42):
    """
    Власна покрокова реалізація RANSAC для оцінки гомографії:
    1. Повторювати N разів:
       - випадково вибрати 4 неполінеарні пари точок;
       - оцінити H_4 (cv2.getPerspectiveTransform);
       - перенести всі точки першого кадру через H_4;
       - знайти inliers, де e_i < tau;
    2. Вибрати H з найбільшою кількістю inliers;
    3. Переоцінити H за всіма inliers через МНК (cv2.findHomography(..., 0)).
    
    Режими (mode):
    - 'fixed': виконувати фіксовану кількість ітерацій max_iters;
    - 'adaptive': динамічно оновлювати N_adapt при покращенні частки узгоджених точок:
      N_adapt = ln(1 - p) / ln(1 - (1 - w)^4), де w = 1 - (inliers / total).
    """
    rng = np.random.default_rng(seed)
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)
    n_points = len(src)

    if n_points < 4:
        raise ValueError("Для обчислення гомографії необхідно щонайменше 4 пари точок.")

    best_H = None
    best_inlier_mask = np.zeros(n_points, dtype=bool)
    max_inliers = 0

    N_current = max_iters if mode == "fixed" else max_iters
    iterations_run = 0

    while iterations_run < N_current:
        iterations_run += 1

        # Випадковий вибір 4 унікальних точок та перевірка невиродженості четвірки
        indices = rng.choice(n_points, size=4, replace=False)
        pts1_sample = src[indices]
        pts2_sample = dst[indices]
        if not _is_valid_quad_sample(pts1_sample, pts2_sample):
            continue

        # Оцінка первинної гомографії за 4 точками
        # getPerspectiveTransform потребує float32 розміром (4, 2)
        try:
            H_candidate = cv2.getPerspectiveTransform(pts1_sample, pts2_sample)
        except cv2.error:
            # У разі виродженої або сингулярної конфігурації
            continue

        if (H_candidate is None or not np.all(np.isfinite(H_candidate))
                or abs(np.linalg.det(H_candidate)) < 1e-9):
            continue

        # Обчислення похибок репроєкції для всіх n_points
        errors = compute_reprojection_errors(src, dst, H_candidate)
        inlier_mask = (errors < tau)
        num_inliers = int(np.sum(inlier_mask))

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_H = H_candidate

            if mode == "adaptive":
                # Оновлення кількості ітерацій за формулою адаптивної зупинки:
                # N = ln(1 - p) / ln(1 - (1 - w)^4), де w = 1 - (inliers / n_points)
                inlier_ratio = num_inliers / n_points
                if inlier_ratio > 0.05:  # оновлюємо лише при виявленні значущої консенсусної вибірки
                    w = 1.0 - inlier_ratio
                    p_all_inliers = (1.0 - w)**4
                    denom = np.log(1.0 - p_all_inliers)
                    if denom < -1e-8:
                        N_adapt = int(np.ceil(np.log(1.0 - p) / denom))
                        N_current = min(max_iters, max(N_adapt, iterations_run))

    # Крок 3 RANSAC: фінальне переоцінення гомографії за всіма inliers через МНК
    if max_inliers >= 4:
        inlier_src = src[best_inlier_mask].reshape(-1, 1, 2)
        inlier_dst = dst[best_inlier_mask].reshape(-1, 1, 2)
        H_refined, _ = cv2.findHomography(inlier_src, inlier_dst, 0)
        if H_refined is not None:
            best_H = H_refined
            # Повторне визначення inlier_mask за уточненою H
            final_errors = compute_reprojection_errors(src, dst, best_H)
            best_inlier_mask = (final_errors < tau)
            mean_inlier_err = float(np.mean(final_errors[best_inlier_mask])) if np.sum(best_inlier_mask) > 0 else float("inf")
        else:
            mean_inlier_err = float("inf")
    else:
        mean_inlier_err = float("inf")

    return best_H, best_inlier_mask, iterations_run, mean_inlier_err


# =========================================================================
# Побудова панорами та лінійне блендування (Частина D)
# =========================================================================

def feather_weights(h, w):
    """
    Вагова карта для лінійного переходу: w(x, y) = відстань до найближчого краю кадру,
    нормована на [0, 1] (0 на краю, 1 у центрі).
    """
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = 255
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return (dist / dist.max()).astype(np.float32)


def stitch_panorama_sequence(images_bgr, homographies_to_ref):
    """
    Зшивання послідовності зображень у єдину панораму:
    1. Розрахунок глобальних меж полотна та внесення трансляційного зсуву T;
    2. Проєкційне перенесення кожного кадру (cv2.warpPerspective);
    3. Плавний лінійний перехід (distance transform / feathering) у зонах перекриття.
    """
    n_frames = len(images_bgr)
    h_orig, w_orig = images_bgr[0].shape[:2]
    corners_orig = np.float32([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]]).reshape(-1, 1, 2)

    all_corners = []
    for H in homographies_to_ref:
        warped_c = cv2.perspectiveTransform(corners_orig, H)
        all_corners.append(warped_c)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    # Трансляційний зсув T, що переводить мінімальні координати в (0, 0)
    tx = -int(x_min)
    ty = -int(y_min)
    T_offset = np.array([
        [1.0, 0.0, float(tx)],
        [0.0, 1.0, float(ty)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    canvas_w = int(x_max - x_min) + 1
    canvas_h = int(y_max - y_min) + 1

    # Акумулятори для блендування з плавною ваговою функцією
    accum_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    accum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # Вагова функція з лінійним спадом до країв кадру (feathering).
    # УВАГА: distanceTransform рахує відстань до найближчого НУЛЬОВОГО пікселя. Якщо маска
    # не містить нулів (суцільні 255), результат -- константа FLT_MAX, і після нормалізації
    # всі ваги дорівнюють 1, тобто замість лінійного переходу виходить просте усереднення
    # з видимими рамками кадрів. Тому маску обрамлюємо нульовою рамкою шириною 1 пк.
    dist_map = feather_weights(h_orig, w_orig)

    for img, H in zip(images_bgr, homographies_to_ref):
        H_canvas = T_offset @ H

        # Варпінг зображення та вагової маски
        warped_img = cv2.warpPerspective(img, H_canvas, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped_weight = cv2.warpPerspective(dist_map, H_canvas, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Додавання до акумулятора
        for c in range(3):
            accum_image[:, :, c] += warped_img[:, :, c].astype(np.float32) * warped_weight
        accum_weight += warped_weight

    # Нормалізація
    valid_mask = accum_weight > 1e-4
    panorama = np.zeros_like(accum_image, dtype=np.uint8)
    for c in range(3):
        panorama[:, :, c] = np.where(valid_mask, np.clip(accum_image[:, :, c] / np.maximum(accum_weight, 1e-4), 0, 255), 0).astype(np.uint8)

    # Кадрування порожніх чорних смуг по краях
    non_zero_y, non_zero_x = np.where(valid_mask)
    if len(non_zero_y) > 0:
        crop_ymin, crop_ymax = non_zero_y.min(), non_zero_y.max()
        crop_xmin, crop_xmax = non_zero_x.min(), non_zero_x.max()
        panorama_cropped = panorama[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1]
    else:
        panorama_cropped = panorama

    return panorama_cropped, panorama, (canvas_w, canvas_h), (tx, ty)
