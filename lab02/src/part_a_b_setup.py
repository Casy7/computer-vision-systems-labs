"""Лабораторна робота №2.
Частини A та B: Параметри обладнання, шаблон та аналіз набору кадрів (Кроки 1-5).
"""
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

from common import DATA, RESULTS, FIGURES, PATTERN_SIZE, SQUARE_SIZE, imread, rgb, save_csv

print("=== Частини A та B: Аналіз набору кадрів та просторового покриття ===")

# Збирання всіх 25 файлів за категоріями
groups = [
    ("frontal", sorted(glob.glob(os.path.join(DATA, "frontal", "*.jpg")))),
    ("angles", sorted(glob.glob(os.path.join(DATA, "angles", "*.jpg")))),
    ("corners", sorted(glob.glob(os.path.join(DATA, "corners", "*.jpg"))))
]

all_files = []
for grp, flist in groups:
    for f in flist:
        all_files.append((grp, f))

print(f"Знайдено всього файлів: {len(all_files)}")

# 1. Вилучення метаданих EXIF та детекція кутів для аналізу покриття
frame_records = []
hulls = []
all_detected_corners = []

for grp, fpath in all_files:
    fname = os.path.basename(fpath)
    rel_path = os.path.relpath(fpath, DATA).replace("\\", "/")
    
    # Читання EXIF через PIL
    pil_img = Image.open(fpath)
    exif = pil_img._getexif() or {}
    exif_map = {TAGS.get(k, k): v for k, v in exif.items()}
    
    model = exif_map.get("Model", "Pixel 5")
    f_mm = float(exif_map.get("FocalLength", 4.38))
    f_35 = float(exif_map.get("FocalLengthIn35mmFilm", 25))
    f_num = float(exif_map.get("FNumber", 1.73))
    exp_time = float(exif_map.get("ExposureTime", 0.04))
    iso = int(exif_map.get("ISOSpeedRatings", 100))
    orient = int(exif_map.get("Orientation", 1))
    
    # Читання в системі сенсора через OpenCV
    img = imread(fpath)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)
    if ret:
        pts = corners.reshape(-1, 2)
        all_detected_corners.append(pts)
        hull = cv2.convexHull(pts.astype(np.float32))
        hulls.append(hull)
        area = cv2.contourArea(hull)
        area_pct = (area / (w * h)) * 100.0
        center_x = float(np.mean(pts[:, 0]))
        center_y = float(np.mean(pts[:, 1]))
    else:
        area_pct = 0.0
        center_x, center_y = 0.0, 0.0

    frame_records.append({
        "group": grp,
        "filename": fname,
        "rel_path": rel_path,
        "width": w,
        "height": h,
        "f_mm": f_mm,
        "f_35": f_35,
        "f_num": f_num,
        "exp_s": f"1/{int(round(1.0/exp_time))}",
        "iso": iso,
        "orientation_tag": orient,
        "detected": 1 if ret else 0,
        "area_pct": round(area_pct, 2),
        "center_x": round(center_x, 1),
        "center_y": round(center_y, 1)
    })

# Збереження CSV списку кадрів
csv_header = [
    "Група", "Файл", "Шлях", "Ширина", "Висота", "F_мм", "F_35мм", "Діафрагма",
    "Витримка", "ISO", "EXIF_Orient", "Знайдено", "Площа_pct", "Центр_X", "Центр_Y"
]
csv_rows = [
    [
        r["group"], r["filename"], r["rel_path"], r["width"], r["height"],
        r["f_mm"], r["f_35"], r["f_num"], r["exp_s"], r["iso"],
        r["orientation_tag"], r["detected"], r["area_pct"], r["center_x"], r["center_y"]
    ]
    for r in frame_records
]
save_csv(os.path.join(RESULTS, "dataset_frames.csv"), csv_header, csv_rows)
print("Збережено dataset_frames.csv")

# 2. Побудова контактного аркуша (5x5 сітка)
fig, axes = plt.subplots(5, 5, figsize=(15, 12))
axes = axes.flatten()

for idx, (grp, fpath) in enumerate(all_files):
    ax = axes[idx]
    img = imread(fpath)
    # Зменшуємо для швидкості побудови огляду
    thumb = cv2.resize(img, (400, 300))
    gray_th = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    ret_th, corners_th = cv2.findChessboardCorners(gray_th, PATTERN_SIZE)
    if ret_th:
        cv2.drawChessboardCorners(thumb, PATTERN_SIZE, corners_th, ret_th)
    
    ax.imshow(rgb(thumb))
    fname = os.path.basename(fpath)
    ax.set_title(f"{fname}\n({grp})", fontsize=8, pad=2)
    ax.axis("off")

plt.subplots_adjust(wspace=0.1, hspace=0.3)
fig.savefig(os.path.join(FIGURES, "ab_contact_sheet.png"), dpi=160)
plt.close(fig)
print("Збережено ab_contact_sheet.png")

# 3. Карта просторового покриття сенсора (Sensor Coverage Map)
fig, ax = plt.subplots(figsize=(8, 6))

# Полотно сенсора 4032 x 3024
ax.set_xlim(0, 4032)
ax.set_ylim(3024, 0) # інверсія Y як у системі координат зображення
ax.set_facecolor("#f8f9fa")

# Рамка сенсора та оптичний центр (2016, 1512)
ax.axvline(2016, color="gray", linestyle="--", alpha=0.5, label="Геометричний центр")
ax.axhline(1512, color="gray", linestyle="--", alpha=0.5)

# Кольорове кодування для трьох груп
group_colors = {
    "frontal": ("#2b5c8f", 0.35, "Фронтальні (масштаб, 5 кадрів)"),
    "angles": ("#d95f02", 0.30, "Під кутом 20-45° (10 кадрів)"),
    "corners": ("#7570b3", 0.30, "Кути та периферія (10 кадрів)")
}

# Побудова полігонів шаблонів
added_labels = set()
for r, hull in zip(frame_records, hulls):
    grp = r["group"]
    color, alpha, label = group_colors[grp]
    poly = plt.Polygon(hull[:, 0, :], facecolor=color, edgecolor=color, alpha=alpha,
                       label=label if grp not in added_labels else None)
    ax.add_patch(poly)
    added_labels.add(grp)

# Відображення центрів шаблонів
for r in frame_records:
    color = group_colors[r["group"]][0]
    ax.plot(r["center_x"], r["center_y"], "o", color=color, markersize=4, alpha=0.8)

ax.set_xlabel("X, піксель")
ax.set_ylabel("Y, піксель")
ax.set_title("Просторове покриття сенсора $4032 \\times 3024$ (25 кадрів)", fontsize=11, pad=8)
ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
ax.grid(True, linestyle=":", alpha=0.6)

fig.savefig(os.path.join(FIGURES, "ab_sensor_coverage.png"), dpi=160)
plt.close(fig)
print("Збережено ab_sensor_coverage.png")

# 4. Підсумкова статистика по групах
group_stats = []
for grp in ["frontal", "angles", "corners"]:
    recs = [r for r in frame_records if r["group"] == grp]
    areas = [r["area_pct"] for r in recs]
    group_stats.append([
        grp,
        len(recs),
        f"{np.mean(areas):.2f}",
        f"{np.min(areas):.2f}",
        f"{np.max(areas):.2f}"
    ])

save_csv(
    os.path.join(RESULTS, "ab_group_stats.csv"),
    ["Група", "Кількість", "Середня_площа_pct", "Мін_площа_pct", "Макс_площа_pct"],
    group_stats
)
print("Збережено ab_group_stats.csv")
print("Частини A та B успішно виконано!\n")

