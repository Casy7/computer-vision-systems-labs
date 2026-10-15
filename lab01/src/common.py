"""Спільні утиліти для Лаб 1."""
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
LAB = os.path.join(ROOT, "lab01")
SHARED_DATA = os.path.join(ROOT, "data")       # спільні вхідні зображення курсу
LAB_DATA = os.path.join(LAB, "data")           # дані, специфічні для лаби (маска, власні кадри)
RESULTS = os.path.join(LAB, "results")
FIGURES = os.path.join(LAB, "figures")
for d in (LAB_DATA, RESULTS, FIGURES):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.bbox": "tight"})


def imread(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def rgb(img):
    """BGR -> RGB для matplotlib."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def psnr(a, b):
    """PSNR = 10 log10(255^2 / MSE); для ідентичних зображень inf."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def save_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
