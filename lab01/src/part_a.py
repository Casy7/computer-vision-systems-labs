"""Лаб 1, частина A: подання зображення (кроки 1-5)."""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

from common import *

SRC = os.path.join(LAB_DATA, "A", "photo.jpg")
img = imread(SRC)
log = []

# ---- Крок 1: базові характеристики ----
info = f"shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}"
print(info)
log.append("STEP1 " + info)

# ---- Крок 2: BGR vs RGB ----
fig, ax = plt.subplots(1, 2, figsize=(10, 2.6))
ax[0].imshow(img)
ax[0].set_title("imshow без перетворення (BGR прочитано як RGB)")
ax[1].imshow(rgb(img))
ax[1].set_title("після cv2.COLOR_BGR2RGB")
for a in ax:
    a.set_xlabel("x, піксель")
    a.set_ylabel("y, піксель")
fig.savefig(os.path.join(FIGURES, "a_bgr_vs_rgb.png"))
plt.close(fig)

# ---- Крок 3: переповнення uint8 ----
wrapped = img + 50                                    # numpy: арифметика за модулем 256
saturated_cv = cv2.add(img, np.full_like(img, 50))    # насичення до 255
saturated_np = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
assert np.array_equal(saturated_cv, saturated_np)
# cv2.add(img, 50) із голим скаляром додає 50 лише до першого каналу (B) - окрема пастка
naive_scalar = cv2.add(img, 50)
scalar_diff = int(np.count_nonzero(np.any(naive_scalar != saturated_cv, axis=2)))

was_bright = np.any(img > 205, axis=2)                # хоч один канал мав переповнитися
became_dark = wrapped.max(axis=2) < img.max(axis=2)   # максимум каналу зменшився
flip = was_bright & became_dark
# піксель, що був яскравим (яскравість >= 200) і став темним (< 100)
gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY)
bright_to_dark = (gray0 >= 200) & (gray1 < 100)
n_wrap_any = int(flip.sum())
n_b2d = int(bright_to_dark.sum())
msg = (f"STEP3 pixels with any channel >205 (overflow): {n_wrap_any} "
       f"({100*n_wrap_any/flip.size:.2f}%); gray>=200 -> gray<100: {n_b2d} "
       f"({100*n_b2d/flip.size:.2f}%); cv2.add(img,50) scalar-only-B differs on {scalar_diff} px")
print(msg)
log.append(msg)

# вікно 256x256 з найбільшою кількістю "білих->темних" пікселів
win = 256
integ = cv2.integral(bright_to_dark.astype(np.uint8))
best, by, bx = -1, 0, 0
for y in range(0, img.shape[0] - win, 32):
    for x in range(0, img.shape[1] - win, 32):
        s = integ[y + win, x + win] - integ[y, x + win] - integ[y + win, x] + integ[y, x]
        if s > best:
            best, by, bx = s, y, x
crop = (slice(by, by + win), slice(bx, bx + win))
log.append(f"STEP3 crop y={by},x={bx},size={win}, bright->dark px in crop={best}")
fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
for a, im, t in zip(ax, (img, wrapped, saturated_cv),
                    ("оригінал", "img + 50 (uint8, переповнення)", "cv2.add / int16+clip")):
    a.imshow(rgb(im[crop]), extent=(bx, bx + win, by + win, by))
    a.set_title(t, fontsize=9)
    a.set_xlabel("x, піксель")
ax[0].set_ylabel("y, піксель")
fig.savefig(os.path.join(FIGURES, "a_overflow.png"))
plt.close(fig)

# ---- Крок 4: стиснення ----
# Основний набір - власний знімок A/photo.jpg. Додатково - кадр B/day_light.png (PNG-контейнер):
# обидва мають JPEG-походження з камери, тому PSNR відображає лише додаткові втрати.
cdir = os.path.join(RESULTS, "compress")
os.makedirs(cdir, exist_ok=True)


def compression_study(src, tag):
    rows = []
    h, w = src.shape[:2]
    p = os.path.join(cdir, f"{tag}_png.png")
    cv2.imwrite(p, src)
    assert np.array_equal(imread(p), src)
    rows.append((tag, "PNG", "-", os.path.getsize(p), "inf", round(os.path.getsize(p) * 8 / (h * w), 3)))
    for q in (95, 75, 50, 25, 10):
        p = os.path.join(cdir, f"{tag}_{q}.jpg")
        cv2.imwrite(p, src, [cv2.IMWRITE_JPEG_QUALITY, q])
        sz = os.path.getsize(p)
        rows.append((tag, "JPEG", q, sz, round(float(psnr(src, imread(p))), 2), round(sz * 8 / (h * w), 3)))
    return rows


second = imread(os.path.join(LAB_DATA, "B", "day_light.png"))
rows = compression_study(img, "photo") + compression_study(second, "day")
save_csv(os.path.join(RESULTS, "a_compression.csv"),
         ["image", "format", "quality", "size_bytes", "psnr_db", "bits_per_pixel"], rows)
for r in rows:
    print(r)

fig, ax = plt.subplots(figsize=(5.5, 3.6))
for tag, lab, st in (("photo", "A/photo.jpg (дерево)", "o-"),
                     ("day", "B/day_light.png (HDD)", "s--")):
    jp = [r for r in rows if r[0] == tag and r[1] == "JPEG"]
    ax.plot([r[3] / 1024 for r in jp], [r[4] for r in jp], st, label=lab)
    for r in jp:
        ax.annotate(f"{r[2]}", (r[3] / 1024, r[4]), textcoords="offset points", xytext=(4, 4), fontsize=7)
ax.set_xscale("log")
ax.set_xlabel("розмір файлу, КіБ (логарифмічна шкала)")
ax.set_ylabel("PSNR, дБ")
ax.grid(True, which="both", alpha=.3)
ax.legend(fontsize=8)
ax.set_title("PSNR від розміру; підписи - якість JPEG", fontsize=9)
fig.savefig(os.path.join(FIGURES, "a_psnr_vs_size.png"))
plt.close(fig)

# фрагмент знімка (гілка на тлі неба): артефакти JPEG
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
cr = (slice(420, 580), slice(330, 530))
for a, (nm, f) in zip(ax, (("оригінал", None), ("JPEG Q=50", "photo_50.jpg"), ("JPEG Q=10", "photo_10.jpg"))):
    im = img if f is None else imread(os.path.join(cdir, f))
    a.imshow(rgb(im[cr]), interpolation="nearest", extent=(cr[1].start, cr[1].stop, cr[0].stop, cr[0].start))
    a.set_title(nm, fontsize=9)
    a.set_xlabel("x, піксель")
ax[0].set_ylabel("y, піксель")
fig.savefig(os.path.join(FIGURES, "a_jpeg_artifacts.png"))
plt.close(fig)

# ---- Крок 5: EXIF ----
Image.MAX_IMAGE_PIXELS = None
wanted = {"FocalLength", "FocalLengthIn35mmFilm", "ExposureTime", "ISOSpeedRatings",
          "PhotographicSensitivity", "Make", "Model", "FNumber"}
candidates = [os.path.join(LAB_DATA, "A", "photo.jpg"),
              os.path.join(LAB_DATA, "B", "day_light.png"),
              os.path.join(LAB_DATA, "B", "lamp_light.png")]
for path in candidates:
    ex = Image.open(path).getexif()
    tags = dict(ex.items())
    tags.update(ex.get_ifd(0x8769))     # Exif IFD
    found = {TAGS.get(k, k): v for k, v in tags.items() if TAGS.get(k) in wanted}
    line = f"STEP5 {os.path.relpath(path, ROOT)}: {found if found else 'EXIF відсутній (немає потрібних тегів)'}"
    print(line)
    log.append(line)

with open(os.path.join(RESULTS, "a_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log) + "\n")
