"""Спільні константи, метадані набору та метрики для Лабораторної роботи №7."""

from pathlib import Path
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data"
PART_A_DIR = DATA_DIR / "part_a"
PART_D_DIR = DATA_DIR / "part_d"
PART_E_DIR = DATA_DIR / "part_e" / "Test"
RESULTS_DIR = LAB_DIR / "results"
FIGURES_DIR = LAB_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "masks_binary").mkdir(parents=True, exist_ok=True)

# Робочий масштаб для Частини B (0.25 від 4032x2268 -> 1008x567 пк)
WORK_SCALE = 0.25

# Опис 10 власних зображень із lab07/data/part_a
DATASET_META = [
    {"id": 1, "name": "Жорсткий диск HDD (підвіконня)", "short": "1. HDD", "category": "Складний"},
    {"id": 2, "name": "Металева пряжка (лакований стіл)", "short": "2. Пряжка", "category": "Складний"},
    {"id": 3, "name": "Карта microSD (стіл, <1% площі)", "short": "3. microSD", "category": "Малий об'єкт"},
    {"id": 4, "name": "Смугаста чашка (шпалери)", "short": "4. Чашка", "category": "Складний"},
    {"id": 5, "name": "Плюшевий Ніко (робочий стіл)", "short": "5. Ніко", "category": "Складний"},
    {"id": 6, "name": "Маска черепа (білий папір)", "short": "6. Маска", "category": "Контрастний"},
    {"id": 7, "name": "Дерев'яний казу (білий папір)", "short": "7. Казу", "category": "Контрастний"},
    {"id": 8, "name": "Білий спорк (червоний фон)", "short": "8. Спорк", "category": "Контрастний"},
    {"id": 9, "name": "Подвійна розетка (сіра стіна)", "short": "9. Розетка", "category": "Контрастний"},
    {"id": 10, "name": "Біла картка (темна тканина)", "short": "10. Картка", "category": "Контрастний"},
]


def load_image_and_mask(img_id: int, scale: float = 1.0):
    """Завантажує кольорове зображення та строго бінарну маску (0/1) за ідентифікатором."""
    img_path = PART_A_DIR / f"{img_id}.png"
    mask_path = PART_A_DIR / f"{img_id}_mask.png"

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask_gray is None:
        raise FileNotFoundError(f"Не вдалося зчитати {img_path} або {mask_path}")

    raw_unique = int(len(np.unique(mask_gray)))
    mask_bin = (mask_gray > 127).astype(np.uint8)

    orig_shape = img.shape[:2]  # (H, W)
    if scale != 1.0:
        new_w = int(round(orig_shape[1] * scale))
        new_h = int(round(orig_shape[0] * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_bin = cv2.resize(mask_bin, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    meta = {
        "orig_h": orig_shape[0],
        "orig_w": orig_shape[1],
        "mask_h": mask_gray.shape[0],
        "mask_w": mask_gray.shape[1],
        "mask_channels": 1 if mask_raw.ndim == 2 else mask_raw.shape[2],
        "raw_unique_levels": raw_unique,
    }
    return img, mask_bin, meta


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Власна реалізація Intersection over Union (IoU) для бінарних масок."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0
    return float(inter / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Власна реалізація коефіцієнта Дайса (F1 за пікселями) для бінарних масок."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int(np.logical_and(p, g).sum())
    denom = int(p.sum() + g.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def compute_pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Обчислює частку правильно класифікованих пікселів (Pixel Accuracy)."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    return float(np.mean(p == g))


def orient_mask_by_border(mask_bin: np.ndarray, border_frac: float = 0.05) -> np.ndarray:
    """
    Автоматично визначає полярність сегментації без використання еталона:
    припускається, що периметр кадру (рамка товщиною 5%) належить фону (0).
    """
    h, w = mask_bin.shape
    bh = max(2, int(round(h * border_frac)))
    bw = max(2, int(round(w * border_frac)))
    border_pixels = np.concatenate([
        mask_bin[:bh, :].ravel(),
        mask_bin[-bh:, :].ravel(),
        mask_bin[bh:-bh, :bw].ravel(),
        mask_bin[bh:-bh, -bw:].ravel(),
    ])
    if float(np.mean(border_pixels > 0)) > 0.5:
        return (1 - (mask_bin > 0).astype(np.uint8)).astype(np.uint8)
    return (mask_bin > 0).astype(np.uint8)
