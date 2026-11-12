"""Спільні утиліти та константи для Лабораторної роботи №5 (Стереопара: ректифікація, диспаратність, глибина)."""

from pathlib import Path
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Налаштування шрифтів для коректного відображення кирилиці
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data"
RESULTS_DIR = LAB_DIR / "results"
FIGURES_DIR = LAB_DIR / "figures"
LAB02_PARAMS = LAB_DIR.parent / "lab02" / "camera_params.yml"
POINTS_SVG = FIGURES_DIR / "points.svg"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Робочий масштаб зображень (0.5 від 4032x2268 -> 2016x1134 пк)
WORK_SCALE = 0.5
# Базова лінія стереопари (Варіант 2: горизонтальний зсув однієї камери B = 10 см = 100 мм)
BASELINE_MM = 100.0
BASELINE_CM = 10.0

# 15 контрольних точок сцени, розмічених користувачем у lab05/figures/points.svg на правому кадрі
# разом із фактичними відстанями Z_meas (см), виміряними рулеткою від площини камери
USER_POINTS_META = {
    1: ("Хвіст емблеми дракона на ширмі", "1. Ширма (дракон)", 119.0),
    2: ("Ніс Блахаджа", "2. Ніс Блахаджа", 131.8),
    3: ("Центр # наліпки C# на ноутбуці", "3. Наліпка C#", 143.0),
    4: ("Центр чайника на малюнку", "4. Малюнок чайника", 259.0),
    5: ("Внутрішній кут рамки телевізора", "5. Рамка ТВ", 251.5),
    6: ("Ручка вікна", "6. Ручка вікна", 313.0),
    7: ("Накінечник олівця", "7. Олівець", 76.8),
    8: ("Емблема рюкзака Deuter", "8. Рюкзак Deuter", 255.0),
    9: ("Логотип фірми кондиціонера", "9. Кондиціонер", 312.0),
    10: ("Шов на подушці", "10. Подушка", 253.0),
    11: ("Внутрішня рамка правої створки ширми", "11. Права створка ширми", 131.0),
    12: ("Кут візерунку на шпалерах в куті кімнати", "12. Шпалери в куті", 348.0),
    13: ("Кут рами вікна", "13. Кут рами вікна", 330.0),
    14: ("Ноутбук, індикатор живлення", "14. Індикатор ноутбука", 140.0),
    15: ("Границя ватмана", "15. Границя ватмана", 106.0),
}


def load_user_svg_points(orig_size=(4032, 2268), scale: float = WORK_SCALE):
    """
    Зчитує координати 15 контрольних точок (point_1 .. point_15) з файла lab05/figures/points.svg
    та переводить їх із системи координат SVG viewBox у пікселі правого кадру при масштабі scale.
    """
    tree = ET.parse(POINTS_SVG)
    root = tree.getroot()
    vb = [float(v) for v in root.attrib["viewBox"].split()]
    vb_w, vb_h = vb[2], vb[3]
    work_w = orig_size[0] * scale
    work_h = orig_size[1] * scale
    sx = work_w / vb_w
    sy = work_h / vb_h

    ns_ink = "{http://www.inkscape.org/namespaces/inkscape}"
    raw_coords = {}
    for elem in root.iter():
        label = elem.attrib.get(ns_ink + "label", "")
        if label.startswith("point_"):
            try:
                idx = int(label.split("_")[1])
            except ValueError:
                continue
            cx = float(elem.attrib["cx"]) * sx
            cy = float(elem.attrib["cy"]) * sy
            raw_coords[idx] = (cx, cy)

    points = []
    for idx in sorted(USER_POINTS_META.keys()):
        if idx not in raw_coords:
            raise KeyError(f"Точку point_{idx} не знайдено у {POINTS_SVG}")
        name, short_name, z_meas_cm = USER_POINTS_META[idx]
        xr, yr = raw_coords[idx]
        points.append({
            "id": idx,
            "name": name,
            "short_name": short_name,
            "xr_orig": float(xr),
            "yr_orig": float(yr),
            "z_meas_cm": float(z_meas_cm),
        })
    return points


def load_stereo_pair(scale: float = WORK_SCALE):
    """Завантажує ліву та праву фотографії стереопари у BGR та Grayscale."""
    left_path = DATA_DIR / "photo_left.jpg"
    right_path = DATA_DIR / "photo_right.jpg"
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(f"Не знайдено файли стереопари у {DATA_DIR}")

    buf_l = np.fromfile(str(left_path), dtype=np.uint8)
    buf_r = np.fromfile(str(right_path), dtype=np.uint8)
    img_l = cv2.imdecode(buf_l, cv2.IMREAD_COLOR)
    img_r = cv2.imdecode(buf_r, cv2.IMREAD_COLOR)

    orig_h, orig_w = img_l.shape[:2]
    if scale != 1.0:
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img_l = cv2.resize(img_l, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_r = cv2.resize(img_r, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    return img_l, img_r, gray_l, gray_r, (orig_w, orig_h)


def load_adapted_intrinsics(orig_size=(4032, 2268), scale: float = WORK_SCALE):
    """
    Завантажує внутрішні параметри камери з Лабораторної роботи №2 (4032x3024, 4:3)
    та адаптує головну точку c_y під центральний вертикальний кроп 16:9 (4032x2268)
    і робочий масштаб scale.
    """
    fs = cv2.FileStorage(str(LAB02_PARAMS), cv2.FILE_STORAGE_READ)
    K_calib = fs.getNode("camera_matrix").mat()
    dist_calib = fs.getNode("distortion_coefficients").mat().flatten()
    calib_w = int(fs.getNode("image_width").real())
    calib_h = int(fs.getNode("image_height").real())
    fs.release()

    orig_w, orig_h = orig_size
    # Врахування симетричного вертикального обрізання при переході 4:3 -> 16:9
    crop_x = (calib_w - orig_w) / 2.0
    crop_y = (calib_h - orig_h) / 2.0

    K_native = K_calib.copy().astype(np.float64)
    K_native[0, 2] -= crop_x
    K_native[1, 2] -= crop_y

    K_scaled = K_native.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale

    info = {
        "calib_size": (calib_w, calib_h),
        "orig_size": (orig_w, orig_h),
        "work_size": (int(round(orig_w * scale)), int(round(orig_h * scale))),
        "crop_offset": (crop_x, crop_y),
        "K_calib": K_calib,
        "K_native": K_native,
        "K_scaled": K_scaled,
        "dist": dist_calib.astype(np.float64),
    }
    return K_scaled, dist_calib.astype(np.float64), info


def detect_and_match_sift(gray_l, gray_r, ratio_thresh: float = 0.75, nfeatures: int = 8000):
    """Виявляє ключові точки SIFT та виконує взаємне зіставлення (Lowe ratio + cross-check)."""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_lr = bf.knnMatch(des_l, des_r, k=2)
    knn_rl = bf.knnMatch(des_r, des_l, k=2)

    good_lr = {}
    for m_n in knn_lr:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_lr[m.queryIdx] = m

    mutual = []
    for m_n in knn_rl:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                q_l = m.trainIdx
                if q_l in good_lr and good_lr[q_l].trainIdx == m.queryIdx:
                    mutual.append(good_lr[q_l])

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in mutual])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in mutual])
    return kp_l, kp_r, mutual, pts_l, pts_r
