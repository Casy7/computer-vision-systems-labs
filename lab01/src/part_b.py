"""Лаб 1, частина B: колірні простори (кроки 6-11).

Вхідні дані (власна зйомка, Google Pixel 5), lab01/data/B:
  day_light.png / day_light_mask.png   - кадр 1: денне світло біля вікна + ручна маска,
  lamp_light.png / lamp_light_mask.png - кадр 2: лампа розжарювання ввечері + ручна маска.
Камеру між кадрами зсунули, тому для кожного кадру є власна еталонна маска.
Об'єкт - жорсткий диск (корпус + зелена плата) на контрастному тлі.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import *

BD = os.path.join(LAB_DATA, "B")


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return m[..., :3].mean(axis=2) > 127          # біле = об'єкт; альфа-канал ігноруємо


f1 = imread(os.path.join(BD, "day_light.png"))
f2 = imread(os.path.join(BD, "lamp_light.png"))
ref1 = load_mask(os.path.join(BD, "day_light_mask.png"))
ref2 = load_mask(os.path.join(BD, "lamp_light_mask.png"))
print("mask pixels", int(ref1.sum()), f"({100*ref1.mean():.1f}%)", int(ref2.sum()), f"({100*ref2.mean():.1f}%)")


def metrics(pred, ref):
    acc = float((pred == ref).mean())
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return acc, float(inter / union)


def in_range(img, lo, hi):
    return cv2.inRange(img, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0


def tune(img, ref, vmax, step=2, rounds=3, pct=(2, 98), stride=2):
    """Підбір меж [lo, hi] по 3 каналах: старт - перцентилі пікселів об'єкта,
    далі покоординатний підйом за IoU (стала процедура для обох просторів)."""
    sm, sr = img[::stride, ::stride], ref[::stride, ::stride]
    px = img[ref]
    lo = [int(np.percentile(px[:, c], pct[0])) for c in range(3)]
    hi = [int(np.percentile(px[:, c], pct[1])) for c in range(3)]
    best = metrics(in_range(sm, lo, hi), sr)[1]
    for _ in range(rounds):
        for c in range(3):
            for is_lo in (True, False):
                cur = lo[c] if is_lo else hi[c]
                for v in range(0, vmax[c] + 1, step):
                    l2, h2 = list(lo), list(hi)
                    if is_lo:
                        l2[c] = v
                    else:
                        h2[c] = v
                    if l2[c] > h2[c]:
                        continue
                    s = metrics(in_range(sm, l2, h2), sr)[1]
                    if s > best:
                        best, cur = s, v
                if is_lo:
                    lo[c] = cur
                else:
                    hi[c] = cur
    return lo, hi


results, bounds, preds = [], {}, {}
spaces = {
    "RGB": (lambda im: im, (255, 255, 255), ("B", "G", "R")),     # масив у порядку BGR
    "HSV": (lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2HSV), (179, 255, 255), ("H", "S", "V")),
}
for name, (conv, vmax, chn) in spaces.items():
    a, b = conv(f1), conv(f2)
    lo, hi = tune(a, ref1, vmax, stride=4)            # межі підбираємо лише на кадрі 1
    lo2, hi2 = tune(b, ref2, vmax, stride=4)          # довідково: межі, підібрані окремо під кадр 2
    p1, p2 = in_range(a, lo, hi), in_range(b, lo, hi) # ті самі межі застосовуємо до кадру 2
    p2o = in_range(b, lo2, hi2)
    acc1, iou1 = metrics(p1, ref1)
    acc2, iou2 = metrics(p2, ref2)
    acc2o, iou2o = metrics(p2o, ref2)
    results.append((name, round(acc1, 4), round(iou1, 4), round(acc2, 4), round(iou2, 4), round(iou2o, 4)))
    bounds[name] = (chn, lo, hi, lo2, hi2)
    preds[name] = (p1, p2)
    print(name, "lo", lo, "hi", hi, "| retuned f2", lo2, hi2)
    print("   frame1 acc/IoU", round(acc1, 4), round(iou1, 4), " frame2 acc/IoU", round(acc2, 4), round(iou2, 4),
          " retuned f2 IoU", round(iou2o, 4))

save_csv(os.path.join(RESULTS, "b_color_spaces.csv"),
         ["space", "acc_frame1", "iou_frame1", "acc_frame2", "iou_frame2", "iou_frame2_retuned"], results)
# межі: RGB у звичному порядку R,G,B
brows = []
for name, (chn, lo, hi, lo2, hi2) in bounds.items():
    if name == "RGB":
        chn, lo, hi, lo2, hi2 = ("R", "G", "B"), lo[::-1], hi[::-1], lo2[::-1], hi2[::-1]
    for i, c in enumerate(chn):
        brows.append((name, c, lo[i], hi[i], lo2[i], hi2[i]))
save_csv(os.path.join(RESULTS, "b_bounds.csv"),
         ["space", "channel", "lo_tuned_f1", "hi_tuned_f1", "lo_retuned_f2", "hi_retuned_f2"], brows)

# базова "тривіальна" оцінка: усе - фон; частка об'єкта в кожному кадрі
with open(os.path.join(RESULTS, "b_log.txt"), "w", encoding="utf-8") as f:
    f.write(f"object_share_f1={ref1.mean():.4f}\nobject_share_f2={ref2.mean():.4f}\n"
            f"all_background_accuracy_f1={(~ref1).mean():.4f}\nall_background_accuracy_f2={(~ref2).mean():.4f}\n")

# ---- рисунки ----
def show(ax, im, title, gray=False):
    ax.imshow(im if gray else rgb(im), cmap="gray" if gray else None, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


fig, ax = plt.subplots(2, 3, figsize=(10, 4.6))
show(ax[0, 0], f1, "кадр 1 (день)")
show(ax[0, 1], f2, "кадр 2 (лампа розжарювання)")
show(ax[0, 2], (ref1 * 255).astype(np.uint8), "еталонна маска кадру 1", gray=True)
for j, (nm, (p1, p2)) in enumerate(preds.items()):
    show(ax[1, j], (p1 * 255).astype(np.uint8),
         f"{nm}: кадр 1, IoU={results[j][2]:.2f}", gray=True)
show(ax[1, 2], (preds["RGB"][1] * 255).astype(np.uint8),
     f"RGB: кадр 2, IoU={results[0][4]:.2f}", gray=True)
fig.savefig(os.path.join(FIGURES, "b_masks_1.png"))
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
show(ax[0], (preds["HSV"][0] * 255).astype(np.uint8), f"HSV: кадр 1, IoU={results[1][2]:.2f}", gray=True)
show(ax[1], (preds["HSV"][1] * 255).astype(np.uint8), f"HSV: кадр 2, IoU={results[1][4]:.2f}", gray=True)
fig.savefig(os.path.join(FIGURES, "b_masks_hsv.png"))
plt.close(fig)

# гістограми H та S пікселів об'єкта в обох кадрах (лог. шкала по осі Y)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.9))
h1, h2 = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV), cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)
for c, (nm, mx) in enumerate((("H, одиниці OpenCV (0-179)", 180), ("S (0-255)", 256))):
    for im, rf, lab in ((h1, ref1, "кадр 1"), (h2, ref2, "кадр 2")):
        hist = np.bincount(im[..., c][rf].ravel(), minlength=mx)
        ax[c].plot(np.arange(mx), hist, label=lab)
    ax[c].set_yscale("log")
    ax[c].set_xlabel(nm)
    ax[c].set_ylabel("кількість пікселів об'єкта")
    ax[c].grid(alpha=.3)
    ax[c].legend(fontsize=8)
fig.savefig(os.path.join(FIGURES, "b_hsv_hist.png"))
plt.close(fig)

# зсув середнього кольору об'єкта
mean1 = f1[ref1].mean(0)[::-1]
mean2 = f2[ref2].mean(0)[::-1]
print("mean RGB object frame1", mean1.round(1), "frame2", mean2.round(1))
with open(os.path.join(RESULTS, "b_log.txt"), "a", encoding="utf-8") as f:
    f.write(f"mean_RGB_f1={mean1.round(1).tolist()}\nmean_RGB_f2={mean2.round(1).tolist()}\n")
