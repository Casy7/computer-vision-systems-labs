"""
Спільні утиліти та константи для Лабораторної роботи № 6:
«Оптичний потік і відстеження об'єктів».
"""

from pathlib import Path
import csv
import numpy as np
import cv2
import matplotlib

# Фіксація випадкового зерна для відтворюваності результатів
np.random.seed(42)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Налаштування шрифтів для коректного відображення кирилиці
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 180,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data"
RESULTS_DIR = LAB_DIR / "results"
FIGURES_DIR = LAB_DIR / "figures"
REPORT_DIR = LAB_DIR / "report"

for d in (RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

VIDEO_A_PATH = DATA_DIR / "one_object.mp4"
VIDEO_B_PATH = DATA_DIR / "two_objects.mp4"

# Полнокадровий формат 4:3 (2592x1944) без чорних бічних смуг pillarbox
FRAME_W = 2592
FRAME_H = 1944
ACTIVE_X0 = 0
ACTIVE_X1 = 2592


def save_csv(path, header, rows):
    """Збереження табличних даних у CSV із кодуванням UTF-8."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def load_video_frames(path):
    """Зчитує всі кадри відео та метадані (frames, fps, w, h)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не вдалося відкрити відеофайл: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, w, h


def build_clean_background(frames, mode="a"):
    """
    Формування чистої моделі фону сцени для відділення рухомих об'єктів та теплого старту MOG2:
      - mode == 'a' (one_object.mp4): часова медіана за 25 рівномірно рознесеними кадрами;
      - mode == 'b' (two_objects.mp4): безшовне поєднання медіани ранніх кадрів (f=0..7)
        і пізніх кадрів (f=last-12..last), де об'єкти не перетинають відповідні половини сцени.
    """
    n = len(frames)
    if mode == "a":
        idxs = np.linspace(0, n - 1, min(25, n), dtype=int)
        stack = np.stack([frames[i] for i in idxs], axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    bg_early = np.median(np.stack(frames[:min(8, n)], axis=0), axis=0).astype(np.float32)
    bg_late = np.median(np.stack(frames[max(0, n - 12):], axis=0), axis=0).astype(np.float32)
    h, w = frames[0].shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    # У перших кадрах обидва об'єкти знаходяться у зоні x in [180, 1500], y in [680, 1300]
    mask_late = ((xx >= 180) & (xx <= 1500) & (yy >= 680) & (yy <= 1300)).astype(np.float32)
    mask_late = cv2.GaussianBlur(mask_late, (51, 51), 15.0)[..., None]
    bg = np.clip(bg_late * mask_late + bg_early * (1.0 - mask_late), 0, 255).astype(np.uint8)
    return bg


def active_roi_mask(h=FRAME_H, w=FRAME_W, margin=10):
    """Бінарна маска робочої області кадру з відступом margin пк від країв."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[margin:h - margin, margin:w - margin] = 255
    return mask


def make_video_writer(path, fps, frame_size):
    """Створює cv2.VideoWriter із кодеком mp4v (або резервним avc1/XVID)."""
    path = Path(path)
    for codec in ("mp4v", "avc1", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), frame_size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Не вдалося створити VideoWriter для {path}")
