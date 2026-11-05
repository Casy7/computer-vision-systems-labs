"""
Лабораторна робота № 3. Частина D.
Експериментальне дослідження повторюваності детекторів (Кроки 14–17).
Досліджуються: Harris, SIFT, ORB за відомих перетворень:
поворот (0-90°), масштаб (0.5-2.0), яскравість (0.5-1.5), шум (sigma 0-20).
"""
import os
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    DETECTOR_COLORS, DETECTOR_MARKERS, DETECTOR_LINESTYLES,
    imread_gray, detect_custom_harris, detect_sift_keypoints, detect_orb_keypoints,
    transform_rotation, transform_scale, transform_brightness, transform_gaussian_noise,
    compute_repeatability, save_csv
)

print("=== Частина D: Експерименти на повторюваність (Repeatability) ===")

# Список тестових зображень
image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
print(f"Знайдено зображень: {len(image_paths)} ({[os.path.basename(p) for p in image_paths]})")

# Словник детекторів
DETECTORS = {
    "Harris": detect_custom_harris,
    "SIFT":   detect_sift_keypoints,
    "ORB":    detect_orb_keypoints
}

# Сітка експериментів (Крок 16)
EXPERIMENTS = {
    "rotation": {
        "param_name": "Поворот, град",
        "values": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "fn": transform_rotation,
        "xlabel": "Кут повороту $\\theta$, градуси",
        "title": "Повторюваність за обертання кадру"
    },
    "scale": {
        "param_name": "Масштаб, коеф",
        "values": [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
        "fn": transform_scale,
        "xlabel": "Масштабний множник $s$",
        "title": "Повторюваність за зміни просторового масштабу"
    },
    "brightness": {
        "param_name": "Яскравість, коеф",
        "values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "fn": transform_brightness,
        "xlabel": "Множник яскравості",
        "title": "Повторюваність за зміни освітленості"
    },
    "noise": {
        "param_name": "Шум_sigma",
        "values": [0, 5, 10, 15, 20],
        "fn": transform_gaussian_noise,
        "xlabel": "СКО адитивного гауссового шуму $\\sigma$",
        "title": "Повторюваність за наявності адитивного шуму"
    }
}

all_results = {}

# Попереднє завантаження зображень та кешування базових точок
print("\nПопереднє кешування ключових точок базових зображень...")
cached_images = {}
for p in image_paths:
    fname = os.path.basename(p)
    gray = imread_gray(p)
    cached_pts = {}
    cached_pts["Harris"], _, _ = detect_custom_harris(gray, max_points=1000, radius=10)
    cached_pts["SIFT"], _, _ = detect_sift_keypoints(gray, max_points=1000, radius=10, min_size=5.0)
    cached_pts["ORB"], _, _ = detect_orb_keypoints(gray, max_points=1000, radius=10)
    cached_images[fname] = {
        "gray": gray,
        "shape": gray.shape,
        "pts": cached_pts
    }
    print(f"  {fname}: Harris={len(cached_pts['Harris'])}, SIFT={len(cached_pts['SIFT'])}, ORB={len(cached_pts['ORB'])}")

for exp_key, exp_cfg in EXPERIMENTS.items():
    print(f"\n--- Проведення тесту: {exp_key.upper()} ---")
    all_results[exp_key] = {det: {val: [] for val in exp_cfg["values"]} for det in DETECTORS}
    csv_rows = []
    for val in exp_cfg["values"]:
        row_data = [val]
        for det_name in DETECTORS:
            rep_list = []
            for fname, img_data in cached_images.items():
                gray = img_data["gray"]
                pts_orig = img_data["pts"][det_name]
                warped, H = exp_cfg["fn"](gray, val)

                # Детектування на перетвореному кадрі
                if det_name == "Harris":
                    pts_warped, _, _ = detect_custom_harris(warped, max_points=1000, radius=10)
                elif det_name == "SIFT":
                    # Для масштабу враховуємо масштабування порогу розміру структурних ознак
                    s_filt = 5.0 * val if exp_key == "scale" else 5.0
                    pts_warped, _, _ = detect_sift_keypoints(warped, max_points=1000, radius=10, min_size=s_filt)
                elif det_name == "ORB":
                    pts_warped, _, _ = detect_orb_keypoints(warped, max_points=1000, radius=10)

                res = compute_repeatability(pts_orig, pts_warped, H, gray.shape, shape2=warped.shape, eps=3.0, margin=15)
                rep = res["repeatability"]
                rep_list.append(rep)
                all_results[exp_key][det_name][val].append(rep)

            mean_rep = float(np.mean(rep_list))
            std_rep = float(np.std(rep_list))
            row_data.extend([f"{mean_rep:.2f}", f"{std_rep:.2f}"])
        csv_rows.append(row_data)
        print(f"  {exp_cfg['param_name']} = {val:5.2f} | "
              f"Harris: {float(row_data[1]):5.1f}% | "
              f"SIFT: {float(row_data[3]):5.1f}% | "
              f"ORB: {float(row_data[5]):5.1f}%")

    csv_header = [
        exp_cfg["param_name"],
        "Harris_Mean_pct", "Harris_Std_pct",
        "SIFT_Mean_pct", "SIFT_Std_pct",
        "ORB_Mean_pct", "ORB_Std_pct"
    ]
    save_csv(os.path.join(RESULTS_DIR, f"d_repeatability_{exp_key}.csv"), csv_header, csv_rows)
    print(f"Збережено d_repeatability_{exp_key}.csv")

# =========================================================================
# Побудова 4 окремих наукових графіків (Крок 16)
# =========================================================================

for exp_key, exp_cfg in EXPERIMENTS.items():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    vals = exp_cfg["values"]

    for det_name in DETECTORS:
        means = [float(np.mean(all_results[exp_key][det_name][v])) for v in vals]
        stds = [float(np.std(all_results[exp_key][det_name][v])) for v in vals]

        ax.plot(
            vals, means,
            label=f"{det_name}",
            color=DETECTOR_COLORS[det_name],
            marker=DETECTOR_MARKERS[det_name],
            linestyle=DETECTOR_LINESTYLES[det_name],
            linewidth=1.8,
            markersize=6
        )

        # Довірча смуга +/- 1 std
        ax.fill_between(
            vals,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=DETECTOR_COLORS[det_name],
            alpha=0.15
        )

    # Спеціальні контрольні маркери та анотації за умовою
    if exp_key == "rotation":
        sift_30 = float(np.mean(all_results["rotation"]["SIFT"][30]))
        ax.axhline(60.0, color="gray", linestyle=":", linewidth=1.0, label=r"Контроль SIFT $\geq$ 60 %")
        ax.annotate(
            f"SIFT 30°: {sift_30:.1f}%",
            xy=(30, sift_30),
            xytext=(35, sift_30 + 8),
            arrowprops=dict(facecolor="green", arrowstyle="->", lw=1.0),
            fontsize=9, fontweight="bold", color="darkgreen"
        )

    if exp_key == "scale":
        h_05 = float(np.mean(all_results["scale"]["Harris"][0.5]))
        s_05 = float(np.mean(all_results["scale"]["SIFT"][0.5]))
        ax.annotate(
            f"SIFT ($s=0.5$): {s_05:.1f}%\n(Гарріс: {h_05:.1f}%)",
            xy=(0.5, s_05),
            xytext=(0.6, s_05 + 10),
            arrowprops=dict(facecolor="darkgreen", arrowstyle="->", lw=1.0),
            fontsize=8.5, fontweight="bold", color="darkgreen"
        )

    ax.set_xlabel(exp_cfg["xlabel"], fontsize=10)
    ax.set_ylabel("Повторюваність (Repeatability), %", fontsize=10)
    ax.set_title(exp_cfg["title"], fontsize=11, pad=10)
    ax.set_ylim(-2, 105)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_name = f"d_repeatability_{exp_key}.png"
    fig.savefig(os.path.join(FIGURES_DIR, fig_name), dpi=180)
    plt.close(fig)
    print(f"Збережено {fig_name}")

# =========================================================================
# Побудова зведеного комбінованого графіка (2х2 матриця панелей)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
axes_flat = axes.flatten()

for idx, (exp_key, exp_cfg) in enumerate(EXPERIMENTS.items()):
    ax = axes_flat[idx]
    vals = exp_cfg["values"]

    for det_name in DETECTORS:
        means = [float(np.mean(all_results[exp_key][det_name][v])) for v in vals]
        stds = [float(np.std(all_results[exp_key][det_name][v])) for v in vals]

        ax.plot(
            vals, means,
            label=f"{det_name}",
            color=DETECTOR_COLORS[det_name],
            marker=DETECTOR_MARKERS[det_name],
            linestyle=DETECTOR_LINESTYLES[det_name],
            linewidth=1.6,
            markersize=5
        )
        ax.fill_between(
            vals,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=DETECTOR_COLORS[det_name],
            alpha=0.12
        )

    letters = ["(а)", "(б)", "(в)", "(г)"]
    ax.set_xlabel(exp_cfg["xlabel"], fontsize=9.5)
    ax.set_ylabel("Повторюваність, %", fontsize=9.5)
    ax.set_title(f"{letters[idx]} {exp_cfg['title']}", fontsize=10.5)
    ax.set_ylim(-2, 105)
    ax.legend(loc="best", framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "d_repeatability_overview.png"), dpi=180)
plt.close(fig)
print("Збережено d_repeatability_overview.png")

# Збереження журналу виконання Частини D
log_path = os.path.join(RESULTS_DIR, "d_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ D: ПОВТОРЮВАНІСТЬ ДЕТЕКТОРІВ ===\n")
    f.write(f"Тестових зображень у вибірці: {len(image_paths)}\n")
    f.write("Умови коректності порівняння (Крок 17):\n")
    f.write("  - Фіксована кількість точок: N = 1000 для всіх детекторів\n")
    f.write("  - Ізотропне кругове NMS: r = 10 пікселів\n")
    f.write("  - Поріг зіставлення: eps = 3.0 пікселі\n")
    f.write("  - Врахування сумісного поля зору (margin = 15 px)\n")
    f.write("  - Взаємно-однозначне (one-to-one) жадібне зіставлення\n\n")
    f.write("Підсумкові показники на контрольних точках:\n")
    sift_rot30 = np.mean(all_results["rotation"]["SIFT"][30])
    harris_rot30 = np.mean(all_results["rotation"]["Harris"][30])
    orb_rot30 = np.mean(all_results["rotation"]["ORB"][30])
    f.write(f"  Поворот на 30 градусів: SIFT = {sift_rot30:.2f}%, Harris = {harris_rot30:.2f}%, ORB = {orb_rot30:.2f}%\n")
    f.write(f"  Контрольний критерій (SIFT > 60%): {'ВИКОНАНО' if sift_rot30 >= 60.0 else 'НЕ ВИКОНАНО'}\n\n")

    sift_sc05 = np.mean(all_results["scale"]["SIFT"][0.5])
    harris_sc05 = np.mean(all_results["scale"]["Harris"][0.5])
    orb_sc05 = np.mean(all_results["scale"]["ORB"][0.5])
    f.write(f"  Масштаб 0.5: SIFT = {sift_sc05:.2f}%, Harris = {harris_sc05:.2f}%, ORB = {orb_sc05:.2f}%\n")
    f.write(f"  Контрольний критерій (Harris помітно нижчий за SIFT): {'ВИКОНАНО' if harris_sc05 < sift_sc05 else 'НЕ ВИКОНАНО'}\n\n")

    f.write(f"  Яскравість 0.5: SIFT = {np.mean(all_results['brightness']['SIFT'][0.5]):.2f}%, Harris = {np.mean(all_results['brightness']['Harris'][0.5]):.2f}%, ORB = {np.mean(all_results['brightness']['ORB'][0.5]):.2f}%\n")
    f.write(f"  Шум sigma=20: SIFT = {np.mean(all_results['noise']['SIFT'][20]):.2f}%, Harris = {np.mean(all_results['noise']['Harris'][20]):.2f}%, ORB = {np.mean(all_results['noise']['ORB'][20]):.2f}%\n")

print(f"Збережено {log_path}")
print("Частина D успішно виконана!\n")
