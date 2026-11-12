"""Частина E: Кількісний та візуальний аналіз систематичних помилок стереозіставлення (Крок 14)."""

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from common import (
    RESULTS_DIR,
    FIGURES_DIR,
)

ERROR_ZONES = [
    {
        "id": "I",
        "type": "Однорідні ділянки без текстури",
        "location": "Білий аркуш А4 та чорна поверхня столу",
        "bbox": (940, 965, 680, 155),  # (x, y, w, h)
        "color": "#ff4b4b",
        "cause": "Плоский профіль функції вартості SAD/BT (відсутність градієнтів яскравості)",
        "mitigation": "WLS-постфільтрація, збільшення blockSize, активне ІЧ-підсвічування текстури",
    },
    {
        "id": "II",
        "type": "Повторювані періодичні патерни",
        "location": "Вертикальні складки штори та шпалери-книжки",
        "bbox": (520, 75, 290, 490),
        "color": "#ffa600",
        "cause": "Багатоекстремальність функції вартості з кроком періоду текстури (хибні мінімуми)",
        "mitigation": "Напівглобальна регуляризація SGBM (штрафи P1, P2), ієрархічний coarse-to-fine пошук",
    },
    {
        "id": "III",
        "type": "Межі об'єктів і напівзакриття (occlusions)",
        "location": "Лівий контур ширми D&D та плюшевої акули",
        "bbox": (1085, 290, 175, 620),
        "color": "#00d4ff",
        "cause": "Фон ліворуч від об'єкта видимий лівою камерою, але перекритий для правої камери",
        "mitigation": "Перевірка ліво-правої узгодженості (disp12MaxDiff = 1) та інтерполяція WLS",
    },
    {
        "id": "IV",
        "type": "Відблиски та дзеркальні поверхні",
        "location": "Скло вікна з дзеркальним відображенням",
        "bbox": (205, 100, 245, 415),
        "color": "#c56cf0",
        "cause": "Порушення ламбертівської моделі: віртуальне зображення за склом має інший паралакс",
        "mitigation": "Поляризаційний фільтр на об'єктиві, робастні дескриптори Census / рангові перетворення",
    },
]


def run_part_e():
    print("=== ЧАСТИНА E: Аналіз систематичних помилок стереозіставлення ===")
    rect_l = cv2.imread(str(RESULTS_DIR / "rect_left.png"), cv2.IMREAD_COLOR)
    disp_bm = np.load(RESULTS_DIR / "bm_disparity.npy")
    disp_sgbm = np.load(RESULTS_DIR / "best_disparity.npy")
    disp_wls = np.load(RESULTS_DIR / "best_disparity_wls.npy")

    rows = []
    for z in ERROR_ZONES:
        x, y, w, h = z["bbox"]
        crop_bm = disp_bm[y : y + h, x : x + w]
        crop_sgbm = disp_sgbm[y : y + h, x : x + w]
        crop_wls = disp_wls[y : y + h, x : x + w]

        inv_bm_pct = 100.0 * float(np.mean(crop_bm <= 0.5))
        inv_sgbm_pct = 100.0 * float(np.mean(crop_sgbm <= 0.5))
        valid_sgbm = crop_sgbm[crop_sgbm > 0.5]
        std_sgbm = float(np.std(valid_sgbm)) if len(valid_sgbm) > 0 else 0.0
        diff_wls = float(np.mean(np.abs(crop_wls[crop_sgbm > 0.5] - valid_sgbm))) if len(valid_sgbm) > 0 else 0.0

        row = {
            "id": z["id"],
            "type": z["type"],
            "location": z["location"],
            "bbox_str": f"[{x}:{x+w}, {y}:{y+h}]",
            "inv_bm_pct": inv_bm_pct,
            "inv_sgbm_pct": inv_sgbm_pct,
            "std_sgbm_px": std_sgbm,
            "diff_wls_px": diff_wls,
            "cause": z["cause"],
            "mitigation": z["mitigation"],
        }
        rows.append(row)
        print(
            f"Зона {z['id']} ({z['type']}): пропусків BM={inv_bm_pct:.1f}%, SGBM={inv_sgbm_pct:.1f}%, std={std_sgbm:.1f} пк"
        )

    with open(RESULTS_DIR / "e_error_zones.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "zone_id",
            "zone_type",
            "scene_location",
            "bbox",
            "invalid_bm_pct",
            "invalid_sgbm_pct",
            "disp_std_px",
            "cause",
            "mitigation",
        ])
        for r in rows:
            writer.writerow([
                r["id"],
                r["type"],
                r["location"],
                r["bbox_str"],
                f"{r['inv_bm_pct']:.1f}",
                f"{r['inv_sgbm_pct']:.1f}",
                f"{r['std_sgbm_px']:.1f}",
                r["cause"],
                r["mitigation"],
            ])

    # Візуалізація 4 зон систематичних помилок (Рисунок 11)
    fig = plt.figure(figsize=(15.0, 8.2), dpi=200)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0], hspace=0.28, wspace=0.18)

    ax_rgb = fig.add_subplot(gs[0, 0:2])
    ax_disp = fig.add_subplot(gs[0, 2:4])

    rgb_img = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("а) Лівий ректифікований кадр (проблемні зони I–IV)", fontsize=10.0, fontweight="bold")
    ax_rgb.axis("off")

    masked_d = np.ma.masked_where(disp_sgbm <= 0.5, disp_sgbm)
    cmap = plt.cm.turbo.copy()
    cmap.set_bad(color="#141414")
    im_d = ax_disp.imshow(masked_d, cmap=cmap, vmin=0, vmax=192)
    ax_disp.set_title("б) Карта диспаратності StereoSGBM (чорний колір -- дірки)", fontsize=10.0, fontweight="bold")
    ax_disp.axis("off")
    cbar = fig.colorbar(im_d, ax=ax_disp, fraction=0.046, pad=0.02)
    cbar.set_label("Диспаратність d, пк", fontsize=9.0)

    for z, r in zip(ERROR_ZONES, rows):
        x, y, w, h = z["bbox"]
        ty = (y + h - 18) if z["id"] == "IV" else (y + 32)
        for ax in (ax_rgb, ax_disp):
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2.2, edgecolor=z["color"], facecolor="none", linestyle="-", zorder=5
            )
            ax.add_patch(rect)
            ax.text(
                x + 8,
                ty,
                f"Зона {z['id']}: {r['inv_sgbm_pct']:.1f}% дірок",
                color="white",
                fontsize=8.2,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", edgecolor=z["color"], linewidth=1.5, alpha=0.85),
                zorder=6,
            )

    # Нижній ряд: збільшені фрагменти (RGB ліворуч + SGBM праворуч у кожній панелі)
    for idx, (z, r) in enumerate(zip(ERROR_ZONES, rows)):
        ax_sub = fig.add_subplot(gs[1, idx])
        x, y, w, h = z["bbox"]
        crop_rgb = rgb_img[y : y + h, x : x + w]
        crop_d = disp_sgbm[y : y + h, x : x + w]

        # Перетворимо crop_d у кольорове зображення turbo із чорним фоном для d <= 0.5
        norm_d = np.clip(crop_d / 192.0, 0.0, 1.0)
        col_d = (plt.cm.turbo(norm_d)[:, :, :3] * 255).astype(np.uint8)
        col_d[crop_d <= 0.5] = [20, 20, 20]

        # Якщо фрагмент вертикальний (w < h), склеїмо горизонтально; якщо горизонтальний (w > h), склеїмо вертикально
        if w >= h:
            sep = np.full((4, w, 3), 255, dtype=np.uint8)
            combined = np.vstack([crop_rgb, sep, col_d])
        else:
            sep = np.full((h, 4, 3), 255, dtype=np.uint8)
            combined = np.hstack([crop_rgb, sep, col_d])

        ax_sub.imshow(combined)
        ax_sub.set_title(
            f"Зона {z['id']}: {z['type']}\n(BM: {r['inv_bm_pct']:.1f}%, SGBM: {r['inv_sgbm_pct']:.1f}% пропусків)",
            fontsize=8.6,
            fontweight="bold",
        )
        for spine in ax_sub.spines.values():
            spine.set_edgecolor(z["color"])
            spine.set_linewidth(2.2)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])

    fig.savefig(FIGURES_DIR / "e_systematic_errors.png", bbox_inches="tight")
    plt.close(fig)
    print("Частину E успішно виконано!")


if __name__ == "__main__":
    run_part_e()
