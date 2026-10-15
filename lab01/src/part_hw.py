"""Лаб 1, домашнє завдання: перевірка власного набору з 20 кадрів (lab01/data/Homework).

Для кожного кадру: EXIF (витримка, ISO), середня яскравість, різкість (дисперсія лапласіана) і кількість
FAST-кутів (перевірка "щонайменше кілька десятків дрібних деталей"). Для сусідніх кадрів серії angle -
оцінка перекриття через гомографію між ORB-ознаками (RANSAC). Додатково - перевірка дублікатів за хешем файлів.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

from common import *

HW = os.path.join(LAB_DATA, "Homework")
SETS = (("angle", 10), ("scale", 5), ("light", 5))
orb = cv2.ORB_create(nfeatures=5000)
fast = cv2.FastFeatureDetector_create(threshold=30)   # кількість кутових деталей без обмеження зверху
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def exif(path):
    ex = Image.open(path).getexif()
    t = dict(ex.items())
    t.update(ex.get_ifd(0x8769))
    d = {TAGS.get(k, k): v for k, v in t.items()}
    return float(d.get("ExposureTime", np.nan)), int(d.get("ISOSpeedRatings", 0))


rows, feats, thumbs = [], {}, []
for kind, n in SETS:
    for i in range(1, n + 1):
        name = f"{kind}{i}"
        path = os.path.join(HW, name + ".jpg")
        im = imread(path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        kp, des = orb.detectAndCompute(small, None)
        feats[name] = (kp, des)
        n_fast = len(fast.detect(small, None))
        exp, iso = exif(path)
        lapvar = float(cv2.Laplacian(small, cv2.CV_64F).var())
        rows.append((name, round(1 / exp), iso, round(float(gray.mean()), 1), round(lapvar, 1), n_fast))
        thumbs.append((name, rgb(cv2.resize(im, (320, 180), interpolation=cv2.INTER_AREA))))
        print(rows[-1])

save_csv(os.path.join(RESULTS, "hw_set.csv"),
         ["frame", "exposure_1_over", "iso", "mean_gray", "laplacian_var", "fast_corners"], rows)

# перекриття сусідніх кадрів серії angle: площа перетину проєкції кадру i на кадр i+1
h, w = 2268 // 2, 4032 // 2
frame_poly = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
ov_rows = []
for i in range(1, 10):
    a, b = f"angle{i}", f"angle{i + 1}"
    (ka, da), (kb, db) = feats[a], feats[b]
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:1500]
    src = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    n_in = int(inl.sum()) if inl is not None else 0
    ov = float("nan")
    if H is not None:
        proj = cv2.perspectiveTransform(frame_poly, H).reshape(-1, 2).astype(np.float32)
        area, _ = cv2.intersectConvexConvex(proj, frame_poly.reshape(-1, 2))
        ov = max(0.0, float(area)) / (w * h) * 100
    ov_rows.append((f"{a}-{b}", len(m), n_in, round(ov, 1)))
    print(ov_rows[-1])
save_csv(os.path.join(RESULTS, "hw_overlap.csv"),
         ["pair", "matches", "inliers", "overlap_percent"], ov_rows)

# контактний аркуш усіх 20 кадрів
fig, axs = plt.subplots(4, 5, figsize=(11, 4.9))
for a, (name, im) in zip(axs.ravel(), thumbs):
    a.imshow(im)
    a.set_title(name, fontsize=8)
    a.axis("off")
fig.savefig(os.path.join(FIGURES, "hw_contact_sheet.png"))
plt.close(fig)

# дублікати (однакові файли)
import hashlib
hs = {}
for kind, n in SETS:
    for i in range(1, n + 1):
        name = f"{kind}{i}"
        hs.setdefault(hashlib.md5(open(os.path.join(HW, name + ".jpg"), "rb").read()).hexdigest(), []).append(name)
dups = [v for v in hs.values() if len(v) > 1]
print("duplicates:", dups)
with open(os.path.join(RESULTS, "hw_log.txt"), "w", encoding="utf-8") as f:
    f.write(f"duplicates={dups}\n")
