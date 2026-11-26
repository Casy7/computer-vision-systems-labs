"""Частини A, B, C та D Лабораторної роботи №7: розмітка, порівняння 4 методів сегментації, парадокс точності та узгодженість розмітників."""

import csv
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic

from common import (
    DATASET_META,
    PART_D_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    WORK_SCALE,
    load_image_and_mask,
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    orient_mask_by_border,
)


def segment_otsu(img_bgr: np.ndarray) -> np.ndarray:
    """Крок 3: Глобальне порогування Оцу з автоматичною орієнтацією полярності за межею кадру."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return orient_mask_by_border(thresh)


def segment_watershed(img_bgr: np.ndarray) -> np.ndarray:
    """
    Крок 4: Вододіл із маркерами суворо за 7 підкроками практикуму:
    1) Оцу; 2) морф. відкриття; 3) distanceTransform; 4) поріг 0.5*max -> sure_fg;
    5) дилатація -> sure_bg; 6) connectedComponents; 7) cv2.watershed.
    """
    # 1. Бінаризація Оцу
    binary = (segment_otsu(img_bgr) * 255).astype(np.uint8)

    # 2. Морфологічне відкриття для прибирання шуму
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    if int(np.sum(opening > 0)) == 0:
        opening = binary.copy()

    # 3. Перетворення відстані
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = float(dist_transform.max())
    if max_dist <= 0.0:
        return (opening > 0).astype(np.uint8)

    # 4. Певний передній план (поріг 0.5 від максимуму)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * max_dist, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # 5. Певний фон (дилатація бінарної маски; зона поза sure_bg_area є фоном)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Маркери зв'язних компонент
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 7. Вододіл
    markers_ws = cv2.watershed(img_bgr.copy(), markers)
    mask_ws = (markers_ws > 1).astype(np.uint8)
    return mask_ws


def segment_slic(img_bgr: np.ndarray, n_segments: int = 400, compactness: float = 10.0) -> np.ndarray:
    """
    Крок 5: Суперпікселі SLIC (skimage.segmentation.slic, n_segments=400, compactness=10)
    та об'єднання суперпікселів у два класи за середнім кольором у просторі CIE Lab (K-Means, k=2).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    segments = slic(img_rgb, n_segments=n_segments, compactness=compactness, start_label=0)
    n_labels = int(segments.max()) + 1

    # Обчислення середнього Lab-кольору для кожного суперпікселя
    mean_colors = np.zeros((n_labels, 3), dtype=np.float32)
    counts = np.bincount(segments.ravel(), minlength=n_labels).astype(np.float32)
    counts_safe = np.maximum(counts, 1.0)
    for c in range(3):
        sums = np.bincount(segments.ravel(), weights=img_lab[:, :, c].ravel(), minlength=n_labels)
        mean_colors[:, c] = sums / counts_safe

    # Кластеризація суперпікселів у 2 класи (cv2.kmeans із фіксованим станом)
    cv2.setRNGSeed(42)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels_km, _ = cv2.kmeans(
        mean_colors, 2, None, criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS
    )
    labels_km = labels_km.ravel()

    mask_slic = labels_km[segments].astype(np.uint8)
    return orient_mask_by_border(mask_slic)


def segment_grabcut(img_bgr: np.ndarray, gt_mask: np.ndarray, iters: int = 5, pad: int = 12) -> np.ndarray:
    """
    Крок 6: Сегментація cv2.grabCut із прямокутником навколо об'єкта (5 ітерацій).
    Прямокутник формується навколо об'єкта з відступом pad пікселів.
    """
    h, w = img_bgr.shape[:2]
    ys, xs = np.where(gt_mask > 0)
    x0 = max(1, int(xs.min()) - pad)
    y0 = max(1, int(ys.min()) - pad)
    x1 = min(w - 2, int(xs.max()) + pad)
    y1 = min(h - 2, int(ys.max()) + pad)
    rect = (x0, y0, max(4, x1 - x0), max(4, y1 - y0))

    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
    mask_gc = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return mask_gc


def run_parts_a_b_c_d():
    print("=== ЧАСТИНИ A--D: Розмітка, 4 методи сегментації, парадокс точності, узгодженість ===")
    dataset_rows = []
    per_image_rows = []
    vis_cache = {}

    methods = [
        ("Otsu", "Оцу (THRESH_OTSU)"),
        ("Watershed", "Вододіл із маркерами"),
        ("SLIC", "SLIC (400, c=10) + K-Means"),
        ("GrabCut", "GrabCut (5 ітерацій)"),
    ]
    metrics_by_method = {k: {"iou": [], "dice": [], "time_ms": []} for k, _ in methods}

    for item in DATASET_META:
        idx = item["id"]
        # Перевірка на повному розмірі та збереження строго бінарних масок {0, 255} (Крок 2)
        _, gt_full, meta = load_image_and_mask(idx, scale=1.0)
        cv2.imwrite(str(RESULTS_DIR / "masks_binary" / f"{idx}_mask_bin.png"), (gt_full * 255).astype(np.uint8))
        area_pct_full = 100.0 * float(np.mean(gt_full))
        size_match = (meta["orig_h"] == meta["mask_h"]) and (meta["orig_w"] == meta["mask_w"])

        dataset_rows.append({
            "id": idx,
            "name": item["name"],
            "category": item["category"],
            "resolution": f"{meta['orig_w']}x{meta['orig_h']}",
            "size_match": "Так" if size_match else "Ні",
            "raw_levels": meta["raw_unique_levels"],
            "clean_levels": 2,
            "area_pct": area_pct_full,
        })

        # Робочий масштаб для 4 алгоритмів сегментації (Кроки 3--6)
        img_w, gt_w, _ = load_image_and_mask(idx, scale=WORK_SCALE)

        t0 = time.perf_counter()
        pred_otsu = segment_otsu(img_w)
        t_otsu = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pred_ws = segment_watershed(img_w)
        t_ws = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pred_slic = segment_slic(img_w, n_segments=400, compactness=10.0)
        t_slic = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pred_gc = segment_grabcut(img_w, gt_w, iters=5, pad=12)
        t_gc = (time.perf_counter() - t0) * 1000.0

        preds = {
            "Otsu": (pred_otsu, t_otsu),
            "Watershed": (pred_ws, t_ws),
            "SLIC": (pred_slic, t_slic),
            "GrabCut": (pred_gc, t_gc),
        }

        row_img = {
            "id": idx,
            "name": item["name"],
            "short": item["short"],
            "category": item["category"],
            "area_pct": area_pct_full,
        }
        for m_key, _ in methods:
            p_mask, dt = preds[m_key]
            iou = compute_iou(p_mask, gt_w)
            dice = compute_dice(p_mask, gt_w)
            metrics_by_method[m_key]["iou"].append(iou)
            metrics_by_method[m_key]["dice"].append(dice)
            metrics_by_method[m_key]["time_ms"].append(dt)
            row_img[f"{m_key}_iou"] = iou
            row_img[f"{m_key}_dice"] = dice

        per_image_rows.append(row_img)
        vis_cache[idx] = {
            "img": img_w,
            "gt": gt_w,
            "Otsu": pred_otsu,
            "Watershed": pred_ws,
            "SLIC": pred_slic,
            "GrabCut": pred_gc,
            "row": row_img,
        }
        print(
            f"Кадр {idx:2d} ({item['short']:11s}, площа {area_pct_full:5.2f}%): "
            f"Otsu={row_img['Otsu_iou']:.3f}, WS={row_img['Watershed_iou']:.3f}, "
            f"SLIC={row_img['SLIC_iou']:.3f}, GrabCut={row_img['GrabCut_iou']:.3f}"
        )

    # Збереження таблиці Частини A
    with open(RESULTS_DIR / "a_dataset_masks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "category", "resolution", "size_match", "raw_levels", "clean_levels", "area_pct"])
        for r in dataset_rows:
            writer.writerow([
                r["id"], r["name"], r["category"], r["resolution"],
                r["size_match"], r["raw_levels"], r["clean_levels"], f"{r['area_pct']:.2f}"
            ])

    # Збереження детальної таблиці Частини B (по всіх 10 кадрах)
    with open(RESULTS_DIR / "b_per_image_iou.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "name", "category", "area_pct",
            "otsu_iou", "watershed_iou", "slic_iou", "grabcut_iou",
            "otsu_dice", "watershed_dice", "slic_dice", "grabcut_dice",
        ])
        for r in per_image_rows:
            writer.writerow([
                r["id"], r["name"], r["category"], f"{r['area_pct']:.2f}",
                f"{r['Otsu_iou']:.3f}", f"{r['Watershed_iou']:.3f}",
                f"{r['SLIC_iou']:.3f}", f"{r['GrabCut_iou']:.3f}",
                f"{r['Otsu_dice']:.3f}", f"{r['Watershed_dice']:.3f}",
                f"{r['SLIC_dice']:.3f}", f"{r['GrabCut_dice']:.3f}",
            ])

    # Збереження зведеної таблиці Частини B (Крок 7: середній IoU, std, min, max, середній Dice, час)
    print("\n--- Зведення методів сегментації (Крок 7) ---")
    summary_rows = []
    with open(RESULTS_DIR / "b_methods_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mean_iou", "std_iou", "min_iou", "max_iou", "mean_dice", "std_dice", "mean_time_ms"])
        for m_key, m_label in methods:
            ious = np.array(metrics_by_method[m_key]["iou"], dtype=np.float64)
            dices = np.array(metrics_by_method[m_key]["dice"], dtype=np.float64)
            times = np.array(metrics_by_method[m_key]["time_ms"], dtype=np.float64)
            row_s = {
                "method": m_label,
                "mean_iou": float(np.mean(ious)),
                "std_iou": float(np.std(ious, ddof=1)),
                "min_iou": float(np.min(ious)),
                "max_iou": float(np.max(ious)),
                "mean_dice": float(np.mean(dices)),
                "std_dice": float(np.std(dices, ddof=1)),
                "mean_time_ms": float(np.mean(times)),
            }
            summary_rows.append(row_s)
            writer.writerow([
                row_s["method"],
                f"{row_s['mean_iou']:.3f}",
                f"{row_s['std_iou']:.3f}",
                f"{row_s['min_iou']:.3f}",
                f"{row_s['max_iou']:.3f}",
                f"{row_s['mean_dice']:.3f}",
                f"{row_s['std_dice']:.3f}",
                f"{row_s['mean_time_ms']:.1f}",
            ])
            print(
                f"{m_label:28s}: IoU = {row_s['mean_iou']:.3f} ± {row_s['std_iou']:.3f} "
                f"[{row_s['min_iou']:.3f} .. {row_s['max_iou']:.3f}], "
                f"Dice = {row_s['mean_dice']:.3f}, час = {row_s['mean_time_ms']:.1f} мс"
            )

    # Рисунок 1 (Частина A): Огляд 10 зображень із накладеним контуром ручної маски
    fig, axes = plt.subplots(2, 5, figsize=(15.0, 6.2), dpi=180)
    for idx in range(1, 11):
        ax = axes[(idx - 1) // 5, (idx - 1) % 5]
        img_rgb = cv2.cvtColor(vis_cache[idx]["img"], cv2.COLOR_BGR2RGB).copy()
        gt_m = vis_cache[idx]["gt"]
        contours, _ = cv2.findContours(gt_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_rgb, contours, -1, (0, 255, 80), 4)
        ax.imshow(img_rgb)
        ax.set_title(
            f"{vis_cache[idx]['row']['short']} ({vis_cache[idx]['row']['area_pct']:.1f}%)",
            fontsize=9.5, fontweight="bold",
        )
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "a_dataset_overview.png", bbox_inches="tight")
    plt.close(fig)

    # Рисунок 2 (Частина B): Візуальне порівняння 4 методів на 6 репрезентативних кадрах
    # (3 контрастних: 6, 8, 9 та 3 складних: 2, 4, 5)
    sample_ids = [6, 8, 9, 2, 4, 5]
    fig, axes = plt.subplots(len(sample_ids), 6, figsize=(15.2, 13.5), dpi=170)
    col_titles = ["Оригінал", "Еталон (GT)", "Оцу", "Вододіл", "SLIC + K-Means", "GrabCut (5 іт.)"]
    for c_idx, title_s in enumerate(col_titles):
        axes[0, c_idx].set_title(title_s, fontsize=10.0, fontweight="bold")

    for r_idx, sid in enumerate(sample_ids):
        vc = vis_cache[sid]
        row_d = vc["row"]
        axes[r_idx, 0].imshow(cv2.cvtColor(vc["img"], cv2.COLOR_BGR2RGB))
        axes[r_idx, 0].set_ylabel(row_d["short"], fontsize=9.5, fontweight="bold", rotation=90, labelpad=6)
        axes[r_idx, 1].imshow(vc["gt"], cmap="gray", vmin=0, vmax=1)
        for c_off, m_key in enumerate(["Otsu", "Watershed", "SLIC", "GrabCut"], start=2):
            axes[r_idx, c_off].imshow(vc[m_key], cmap="gray", vmin=0, vmax=1)
            axes[r_idx, c_off].set_xlabel(f"IoU = {row_d[f'{m_key}_iou']:.2f}", fontsize=8.8, fontweight="bold")
        for c_i in range(6):
            axes[r_idx, c_i].set_xticks([])
            axes[r_idx, c_i].set_yticks([])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "b_segmentation_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Рисунок 3 (Частина B): Графік IoU по 10 зображеннях та діаграма розкиду (boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.8), dpi=200, gridspec_kw={"width_ratios": [1.45, 1.0]})
    x_pos = np.arange(1, 11)
    width_b = 0.20
    colors_m = {"Otsu": "#e15759", "Watershed": "#f28e2b", "SLIC": "#59a14f", "GrabCut": "#4e79a7"}
    labels_short = {"Otsu": "Оцу", "Watershed": "Вододіл", "SLIC": "SLIC", "GrabCut": "GrabCut"}
    for j, (m_key, _) in enumerate(methods):
        vals = [r[f"{m_key}_iou"] for r in per_image_rows]
        axes[0].bar(
            x_pos + (j - 1.5) * width_b, vals, width=width_b,
            color=colors_m[m_key], edgecolor="black", linewidth=0.45, label=labels_short[m_key]
        )
    axes[0].axvline(5.5, color="#555555", linestyle="--", linewidth=1.2)
    axes[0].text(3.0, 1.03, "Складний / нерівномірний фон", ha="center", fontsize=8.6, color="#444444", fontweight="bold")
    axes[0].text(8.0, 1.03, "Контрастний однотонний фон", ha="center", fontsize=8.6, color="#444444", fontweight="bold")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([r["short"] for r in per_image_rows], rotation=28, ha="right", fontsize=8.3)
    axes[0].set_ylim(0.0, 1.22)
    axes[0].set_ylabel("Метрика якості IoU", fontsize=9.5)
    axes[0].set_title("а) IoU кожного методу на 10 зображеннях набору", fontsize=10.2, fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.3, ncol=4, loc="upper center", framealpha=0.92)

    box_data = [metrics_by_method[k]["iou"] for k, _ in methods]
    bp = axes[1].boxplot(
        box_data, patch_artist=True, widths=0.52,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.3),
        capprops=dict(linewidth=1.3),
    )
    for patch, (m_key, _) in zip(bp["boxes"], methods):
        patch.set_facecolor(colors_m[m_key])
        patch.set_alpha(0.75)
    # Додаємо точки окремих зображень і середнє
    for j, (m_key, _) in enumerate(methods, start=1):
        vals = metrics_by_method[m_key]["iou"]
        jitter = np.linspace(-0.12, 0.12, len(vals))
        axes[1].scatter(np.full(len(vals), j) + jitter, vals, color="black", s=16, alpha=0.65, zorder=4)
        axes[1].scatter([j], [np.mean(vals)], color="yellow", edgecolor="black", marker="D", s=46, zorder=5,
                        label="Середнє" if j == 1 else None)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xticklabels(["Оцу", "Вододіл", "SLIC", "GrabCut"], fontsize=9.2)
    axes[1].set_ylim(-0.03, 1.08)
    axes[1].set_ylabel("Розподіл IoU по 10 зображеннях", fontsize=9.5)
    axes[1].set_title("б) Розкид (min--max, квартилі, середнє IoU)", fontsize=10.2, fontweight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "b_metrics_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    # --- ЧАСТИНА C (Крок 8): Демонстрація оманливості точності на кадрах із площею < 10% ---
    print("\n--- Частина C: Демонстрація оманливості точності (площа об'єкта < 10%) ---")
    small_ids = [r["id"] for r in dataset_rows if r["area_pct"] < 10.0]
    paradox_rows = []
    for sid in small_ids:
        _, gt_f, _ = load_image_and_mask(sid, scale=1.0)
        empty_mask = np.zeros_like(gt_f, dtype=np.uint8)
        acc_empty = 100.0 * compute_pixel_accuracy(empty_mask, gt_f)
        iou_empty = compute_iou(empty_mask, gt_f)
        dice_empty = compute_dice(empty_mask, gt_f)

        vc = vis_cache[sid]
        acc_gc = 100.0 * compute_pixel_accuracy(vc["GrabCut"], vc["gt"])
        iou_gc = vc["row"]["GrabCut_iou"]
        acc_otsu = 100.0 * compute_pixel_accuracy(vc["Otsu"], vc["gt"])
        iou_otsu = vc["row"]["Otsu_iou"]

        p_row = {
            "id": sid,
            "name": vc["row"]["name"],
            "area_pct": vc["row"]["area_pct"],
            "empty_acc": acc_empty,
            "empty_iou": iou_empty,
            "empty_dice": dice_empty,
            "otsu_acc": acc_otsu,
            "otsu_iou": iou_otsu,
            "gc_acc": acc_gc,
            "gc_iou": iou_gc,
        }
        paradox_rows.append(p_row)
        print(
            f"Кадр #{sid} ({p_row['name']}, площа {p_row['area_pct']:.2f}%): "
            f"Порожня маска -> Acc={acc_empty:.2f}%, IoU={iou_empty:.3f} | "
            f"GrabCut -> Acc={acc_gc:.2f}%, IoU={iou_gc:.3f}"
        )

    with open(RESULTS_DIR / "c_accuracy_paradox.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "name", "area_pct",
            "empty_acc_pct", "empty_iou", "empty_dice",
            "otsu_acc_pct", "otsu_iou", "gc_acc_pct", "gc_iou",
        ])
        for r in paradox_rows:
            writer.writerow([
                r["id"], r["name"], f"{r['area_pct']:.2f}",
                f"{r['empty_acc']:.2f}", f"{r['empty_iou']:.3f}", f"{r['empty_dice']:.3f}",
                f"{r['otsu_acc']:.2f}", f"{r['otsu_iou']:.3f}",
                f"{r['gc_acc']:.2f}", f"{r['gc_iou']:.3f}",
            ])

    # Рисунок 4 (Частина C): Ілюстрація парадоксу точності на кадрі #3 (microSD, 0.75%)
    vc3 = vis_cache[3]
    p3 = [r for r in paradox_rows if r["id"] == 3][0]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.4), dpi=190, gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    axes[0].imshow(cv2.cvtColor(vc3["img"], cv2.COLOR_BGR2RGB))
    axes[0].set_title("а) Кадр №3 (microSD, 0.75%)", fontsize=9.5, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(vc3["gt"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("б) Еталонна маска (GT)", fontsize=9.5, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(np.zeros_like(vc3["gt"]), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"в) Порожня маска\nAcc = {p3['empty_acc']:.2f}%, IoU = 0.000", fontsize=9.2, fontweight="bold")
    axes[2].axis("off")

    # Панель г: Порівняння Accuracy vs IoU для 3 малих об'єктів (#3, #10, #9)
    s_labels = [f"#{r['id']} ({r['area_pct']:.2f}%)" for r in paradox_rows]
    sx = np.arange(len(paradox_rows))
    axes[3].bar(sx - 0.27, [r["empty_acc"] / 100.0 for r in paradox_rows], width=0.25, color="#e15759", edgecolor="black", label="Порожня: Accuracy")
    axes[3].bar(sx, [r["empty_iou"] for r in paradox_rows], width=0.25, color="#222222", edgecolor="black", label="Порожня: IoU (= 0)")
    axes[3].bar(sx + 0.27, [r["gc_iou"] for r in paradox_rows], width=0.25, color="#4e79a7", edgecolor="black", label="GrabCut: IoU")
    for k_i, r in enumerate(paradox_rows):
        axes[3].text(k_i - 0.27, r["empty_acc"] / 100.0 + 0.02, f"{r['empty_acc']:.1f}%", ha="center", fontsize=7.8, fontweight="bold")
        axes[3].text(k_i, 0.03, "0.00", ha="center", fontsize=7.8, fontweight="bold", color="#b00020")
        axes[3].text(k_i + 0.27, r["gc_iou"] + 0.02, f"{r['gc_iou']:.2f}", ha="center", fontsize=7.8, fontweight="bold")
    axes[3].set_xticks(sx)
    axes[3].set_xticklabels(s_labels, fontsize=8.8)
    axes[3].set_ylim(0, 1.32)
    axes[3].set_ylabel("Значення метрики (0..1)", fontsize=9.2)
    axes[3].set_title("г) Accuracy vs IoU за дисбалансу класів", fontsize=9.5, fontweight="bold")
    axes[3].grid(axis="y", linestyle=":", alpha=0.6)
    axes[3].legend(fontsize=7.6, loc="upper center", ncol=1, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "c_accuracy_paradox.png", bbox_inches="tight")
    plt.close(fig)

    # --- ЧАСТИНА D (Кроки 9--10): Узгодженість між розмітниками ---
    colleague_masks = sorted(PART_D_DIR.glob("*.png")) if PART_D_DIR.exists() else []
    with open(RESULTS_DIR / "d_inter_annotator.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "image_id", "annotator1_area_px", "annotator2_area_px", "iou", "dice", "diff_px"])
        if colleague_masks:
            m2_path = colleague_masks[0]
            # Витягуємо номер кадру з імені файлу (наприклад, 5_mask_colleague.png -> 5)
            digits = "".join(ch for ch in m2_path.stem.split("_")[0] if ch.isdigit())
            img_id_d = int(digits) if digits else 1
            _, gt1, _ = load_image_and_mask(img_id_d, scale=1.0)
            m2_raw = cv2.imread(str(m2_path), cv2.IMREAD_GRAYSCALE)
            if m2_raw.shape != gt1.shape:
                m2_raw = cv2.resize(m2_raw, (gt1.shape[1], gt1.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt2 = (m2_raw > 127).astype(np.uint8)
            iou_d = compute_iou(gt1, gt2)
            dice_d = compute_dice(gt1, gt2)
            diff_px = int(np.sum(gt1 != gt2))
            writer.writerow(["DONE", img_id_d, int(gt1.sum()), int(gt2.sum()), f"{iou_d:.4f}", f"{dice_d:.4f}", diff_px])
            print(f"\nЧастина D виконана на кадрі #{img_id_d}: IoU = {iou_d:.4f}, Dice = {dice_d:.4f}")
        else:
            writer.writerow(["PENDING", "-", 0, 0, "0.0000", "0.0000", 0])
            print("\nЧастина D (Кроки 9--10): очікується незалежна маска колеги у lab07/data/part_d/")

    print("Частини A--D опрацьовано!")


if __name__ == "__main__":
    run_parts_a_b_c_d()
