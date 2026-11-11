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

DATA_DIR = os.path.join(LAB_DIR, "data")
DATA_HW_DIR = DATA_DIR
RESULTS_DIR = os.path.join(LAB_DIR, "results")
FIGURES_DIR = os.path.join(LAB_DIR, "figures")
REPORT_DIR = os.path.join(LAB_DIR, "report")

# Єдиний робочий масштаб зображень серії (4032x2268 -> 2016x1134 пк)
WORK_SCALE = 0.5

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


def imread_bgr(path, scale=WORK_SCALE):
    """Зчитування зображення в колірному просторі BGR без автоповоротів EXIF із робочим масштабуванням."""
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"Файл не знайдено або пошкоджено: {path}")
    if scale is not None and scale != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def imread_rgb(path, scale=WORK_SCALE):
    """Зчитування зображення в RGB."""
    return cv2.cvtColor(imread_bgr(path, scale=scale), cv2.COLOR_BGR2RGB)


def imread_gray(path, scale=WORK_SCALE):
    """Зчитування зображення у градаціях сірого."""
    return cv2.cvtColor(imread_bgr(path, scale=scale), cv2.COLOR_BGR2GRAY)


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
       - випадково вибрати 4 неколінеарні пари точок;
       - оцінити H_4 (cv2.getPerspectiveTransform);
       - перенести всі точки першого кадру через H_4;
       - знайти inliers, де e_i < tau;
    2. Вибрати H з найбільшою кількістю inliers;
    3. Переоцінити H за всіма inliers через МНК (cv2.findHomography(..., 0)).
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

    N_current = max_iters
    iterations_run = 0

    while iterations_run < N_current:
        iterations_run += 1

        indices = rng.choice(n_points, size=4, replace=False)
        pts1_sample = src[indices]
        pts2_sample = dst[indices]
        if not _is_valid_quad_sample(pts1_sample, pts2_sample):
            continue

        try:
            H_candidate = cv2.getPerspectiveTransform(pts1_sample, pts2_sample)
        except cv2.error:
            continue

        if (H_candidate is None or not np.all(np.isfinite(H_candidate))
                or abs(np.linalg.det(H_candidate)) < 1e-9):
            continue

        errors = compute_reprojection_errors(src, dst, H_candidate)
        inlier_mask = (errors < tau)
        num_inliers = int(np.sum(inlier_mask))

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_H = H_candidate

            if mode == "adaptive":
                inlier_ratio = num_inliers / n_points
                if inlier_ratio > 0.05:
                    w = 1.0 - inlier_ratio
                    p_all_inliers = (1.0 - w)**4
                    denom = np.log(1.0 - p_all_inliers)
                    if denom < -1e-8:
                        N_adapt = int(np.ceil(np.log(1.0 - p) / denom))
                        N_current = min(max_iters, max(N_adapt, iterations_run))

    if max_inliers >= 4:
        inlier_src = src[best_inlier_mask].reshape(-1, 1, 2)
        inlier_dst = dst[best_inlier_mask].reshape(-1, 1, 2)
        H_refined, _ = cv2.findHomography(inlier_src, inlier_dst, 0)
        if H_refined is not None:
            best_H = H_refined
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

def feather_weights(h, w, valid_mask=None):
    """
    Вагова карта для лінійного переходу: w(x, y) = відстань до найближчого краю
    валідної області кадру, нормована на [0, 1] (0 на краю, 1 у центрі).
    """
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    if valid_mask is None:
        padded[1:-1, 1:-1] = 255
    else:
        padded[1:-1, 1:-1] = (valid_mask.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    dmax = float(dist.max())
    return (dist / max(dmax, 1e-6)).astype(np.float32)


def clip_rect_halfplane(w, h, a, b, c):
    """
    Відсікання прямокутника кадру [0, w] x [0, h] півплощиною a*x + b*y + c >= 0
    за алгоритмом Сазерленда–Ходжмана (повертає опуклий багатокутник у float32).
    """
    poly = [
        np.array([0.0, 0.0]),
        np.array([float(w), 0.0]),
        np.array([float(w), float(h)]),
        np.array([0.0, float(h)])
    ]
    out = []
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        v1 = a * p1[0] + b * p1[1] + c
        v2 = a * p2[0] + b * p2[1] + c
        if v1 >= 0:
            out.append(p1)
        if (v1 >= 0) != (v2 >= 0):
            t = v1 / (v1 - v2)
            out.append(p1 + t * (p2 - p1))
    return np.array(out, dtype=np.float32)


def stitch_panorama_sequence(images_bgr, homographies_to_ref, min_w_rel=0.28, return_polys=False):
    """
    Зшивання послідовності зображень у єдину панораму:
    1. Відсікання областей крайніх кадрів, де проєктивний знаменник w_z наближається
       до лінії горизонту (w_rel < min_w_rel), та розрахунок глобальних меж полотна і зсуву T;
    2. Проєкційне перенесення кожного кадру (cv2.warpPerspective);
    3. Плавний лінійний перехід (distance transform / feathering) у зонах перекриття.
    """
    n_frames = len(images_bgr)
    h_orig, w_orig = images_bgr[0].shape[:2]
    ref_idx = n_frames // 2

    all_corners = []
    valid_polys = []
    norm_H_list = []

    for idx, H in enumerate(homographies_to_ref):
        if idx < ref_idx:
            x_ov = w_orig * 0.70
        elif idx > ref_idx:
            x_ov = w_orig * 0.30
        else:
            x_ov = w_orig * 0.50
        y_ov = h_orig * 0.50
        w_ov = float(H[2, 0] * x_ov + H[2, 1] * y_ov + H[2, 2])
        if abs(w_ov) < 1e-9:
            w_ov = 1.0
        Hn = H / w_ov
        norm_H_list.append(Hn)

        poly = clip_rect_halfplane(w_orig, h_orig, Hn[2, 0], Hn[2, 1], Hn[2, 2] - min_w_rel)
        if len(poly) < 3:
            poly = _frame_corners((h_orig, w_orig))
        valid_polys.append(poly)
        warped_c = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), Hn)
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

    accum_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    accum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for img, Hn, poly in zip(images_bgr, norm_H_list, valid_polys):
        vmask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.fillConvexPoly(vmask, np.round(poly).astype(np.int32), 1)
        dist_map = feather_weights(h_orig, w_orig, valid_mask=(vmask > 0))
        img_masked = img.copy()
        img_masked[vmask == 0] = 0

        H_canvas = T_offset @ Hn
        warped_img = cv2.warpPerspective(img_masked, H_canvas, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped_weight = cv2.warpPerspective(dist_map, H_canvas, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        for c in range(3):
            accum_image[:, :, c] += warped_img[:, :, c].astype(np.float32) * warped_weight
        accum_weight += warped_weight

    valid_mask = accum_weight > 1e-4
    panorama = np.zeros_like(accum_image, dtype=np.uint8)
    for c in range(3):
        panorama[:, :, c] = np.where(valid_mask, np.clip(accum_image[:, :, c] / np.maximum(accum_weight, 1e-4), 0, 255), 0).astype(np.uint8)

    non_zero_y, non_zero_x = np.where(valid_mask)
    if len(non_zero_y) > 0:
        crop_ymin, crop_ymax = non_zero_y.min(), non_zero_y.max()
        crop_xmin, crop_xmax = non_zero_x.min(), non_zero_x.max()
        panorama_cropped = panorama[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1]
    else:
        panorama_cropped = panorama

    if return_polys:
        return panorama_cropped, panorama, (canvas_w, canvas_h), (tx, ty), valid_polys, norm_H_list
    return panorama_cropped, panorama, (canvas_w, canvas_h), (tx, ty)
