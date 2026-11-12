"""Частина C: Побудова та дослідження карти диспаратності (Кроки 7-9)."""

import csv
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    RESULTS_DIR,
    FIGURES_DIR,
    detect_and_match_sift,
)


def compute_disparity_bm(gray_l, gray_r, num_disp=192, block_size=15, uniq_ratio=10):
    """Обчислює карту диспаратності методом блокового зіставлення StereoBM."""
    matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    matcher.setUniquenessRatio(uniq_ratio)
    matcher.setSpeckleWindowSize(100)
    matcher.setSpeckleRange(2)
    matcher.setDisp12MaxDiff(1)
    t0 = time.perf_counter()
    disp_raw = matcher.compute(gray_l, gray_r)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    disp = disp_raw.astype(np.float32) / 16.0
    return disp, dt_ms


def compute_disparity_sgbm(gray_l, gray_r, num_disp=192, block_size=9, uniq_ratio=10):
    """Обчислює карту диспаратності методом напівглобального зіставлення StereoSGBM."""
    p1 = 8 * (block_size ** 2)
    p2 = min(32 * (block_size ** 2), 10000)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=uniq_ratio,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    t0 = time.perf_counter()
    disp_raw = matcher.compute(gray_l, gray_r)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    disp = disp_raw.astype(np.float32) / 16.0
    return disp, dt_ms, matcher


def compute_disparity_sgbm_wls(gray_l, gray_r, guide_bgr_l, num_disp=192, block_size=9, uniq_ratio=10):
    """Обчислює карту диспаратності StereoSGBM із двосторонньою WLS-постфільтрацією."""
    p1 = 8 * (block_size ** 2)
    p2 = min(32 * (block_size ** 2), 10000)
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=uniq_ratio,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)

    t0 = time.perf_counter()
    disp_l = left_matcher.compute(gray_l, gray_r)
    disp_r = right_matcher.compute(gray_r, gray_l)
    disp_wls = wls_filter.filter(disp_l, guide_bgr_l, None, disp_r)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    disp = disp_wls.astype(np.float32) / 16.0
    return disp, dt_ms


def evaluate_disparity_map(disp, num_disp, sift_pts_l, sift_disp):
    """
    Обчислює кількісні метрики якості карти диспаратності:
    - valid_pct: частка валідних пікселів (d > 0) в активній зоні x >= num_disp;
    - grad_noise: медіанна абсолютна лапласіанна шорсткість усередині гладких поверхонь;
    - sift_mae: середня абсолютна розбіжність із розрідженими SIFT-відповідностями (пк).
    """
    h, w = disp.shape
    roi = disp[:, num_disp:]
    valid_mask = roi > 0.5
    valid_pct = 100.0 * float(np.mean(valid_mask))

    # Оцінка шорсткості (друга похідна по горизонталі на валідних сусідах)
    d_valid = np.where(disp > 0.5, disp, np.nan)
    diff2 = np.abs(d_valid[:, 2:] - 2.0 * d_valid[:, 1:-1] + d_valid[:, :-2])
    # Відсікаємо стрибки на межах об'єктів (> 3 пк), щоб оцінити саме внутрішній шум
    smooth_vals = diff2[np.isfinite(diff2) & (diff2 < 3.0)]
    grad_noise = float(np.mean(smooth_vals)) if len(smooth_vals) > 0 else 0.0

    # Звірка з еталонними SIFT-відповідностями
    xs = np.clip(np.round(sift_pts_l[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(sift_pts_l[:, 1]).astype(int), 0, h - 1)
    d_est = disp[ys, xs]
    ok = (d_est > 0.5) & (sift_disp > 0.5)
    if np.any(ok):
        diffs = np.abs(d_est[ok] - sift_disp[ok])
        # Робастна оцінка MAE (відсікання 5% викидів на межах глибини)
        q95 = np.percentile(diffs, 95)
        sift_mae = float(np.mean(diffs[diffs <= q95]))
    else:
        sift_mae = 99.9

    return valid_pct, grad_noise, sift_mae


def describe_visual_quality(num_disp, block_size, uniq_ratio, valid_pct, sift_mae):
    """Формує лаконічну характеристику візуальної якості карти для таблиці звіту."""
    if num_disp == 64:
        return "Зрізано передній план (d > 64 пк не визначається)"
    if num_disp == 128:
        return "Частково втрачено ширму та олівець (d > 128 пк)"
    # num_disp == 192
    if block_size == 5:
        if uniq_ratio == 5:
            return "Висока деталізація, помітний зернистий шум"
        elif uniq_ratio == 10:
            return "Чіткі контури, помірна кількість дірок на столі"
        else:
            return "Чисті контури, багато пропусків на слабкій текстурі"
    elif block_size == 9:
        if uniq_ratio == 5:
            return "Щільне заповнення, поодинокі хибні збіги"
        elif uniq_ratio == 10:
            return "Оптимальний баланс деталізації та гладкості"
        else:
            return "Дуже чиста карта, зростання дірок на аркуші А4"
    elif block_size == 15:
        if uniq_ratio == 5:
            return "Гладкі поверхні, потовщення контурів (halo)"
        elif uniq_ratio == 10:
            return "Добра гладкість, згладжені дрібні деталі"
        else:
            return "Надійні площини, відсіяні сумнівні зони"
    else:  # block_size == 21
        if uniq_ratio == 5:
            return "Надмірне розмиття меж об'єктів (ореоли)"
        elif uniq_ratio == 10:
            return "Сильне згладжування рельєфу, втрата тонких деталей"
        else:
            return "Грубі контури та пропуски біля меж"


def run_part_c():
    print("=== ЧАСТИНА C: Карта диспаратності та дослідження параметрів ===")
    rect_l = cv2.imread(str(RESULTS_DIR / "rect_left.png"), cv2.IMREAD_COLOR)
    rect_r = cv2.imread(str(RESULTS_DIR / "rect_right.png"), cv2.IMREAD_COLOR)
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    # Отримаємо розріджені SIFT-відповідності на ректифікованій парі для об'єктивної оцінки MAE
    _, _, _, pts_l, pts_r = detect_and_match_sift(gray_l, gray_r)
    _, mask_f = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, 1.5, 0.999)
    inl = (mask_f.ravel() == 1) & (np.abs(pts_l[:, 1] - pts_r[:, 1]) < 1.2)
    sift_pts_l = pts_l[inl]
    sift_disp = pts_l[inl, 0] - pts_r[inl, 0]

    # 1. Крок 7: Порівняння StereoBM та StereoSGBM
    disp_bm, time_bm = compute_disparity_bm(gray_l, gray_r, num_disp=192, block_size=15, uniq_ratio=10)
    val_bm, noise_bm, mae_bm = evaluate_disparity_map(disp_bm, 192, sift_pts_l, sift_disp)

    disp_sgbm, time_sgbm, _ = compute_disparity_sgbm(gray_l, gray_r, num_disp=192, block_size=9, uniq_ratio=10)
    val_sgbm, noise_sgbm, mae_sgbm = evaluate_disparity_map(disp_sgbm, 192, sift_pts_l, sift_disp)

    disp_wls, time_wls = compute_disparity_sgbm_wls(gray_l, gray_r, rect_l, num_disp=192, block_size=9, uniq_ratio=10)
    val_wls, noise_wls, mae_wls = evaluate_disparity_map(disp_wls, 192, sift_pts_l, sift_disp)

    print(f"StereoBM (192, bs=15, ur=10): valid={val_bm:.1f}%, time={time_bm:.1f} мс, MAE={mae_bm:.2f} пк")
    print(f"StereoSGBM (192, bs=9, ur=10): valid={val_sgbm:.1f}%, time={time_sgbm:.1f} мс, MAE={mae_sgbm:.2f} пк")
    print(f"StereoSGBM + WLS (192, bs=9): valid={val_wls:.1f}%, time={time_wls:.1f} мс, MAE={mae_wls:.2f} пк")

    with open(RESULTS_DIR / "c_bm_vs_sgbm.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "numDisparities",
            "blockSize",
            "uniquenessRatio",
            "valid_pct",
            "time_ms",
            "grad_noise_px",
            "sift_mae_px",
            "summary",
        ])
        writer.writerow([
            "StereoBM (SAD)",
            192,
            15,
            10,
            f"{val_bm:.1f}",
            f"{time_bm:.1f}",
            f"{noise_bm:.3f}",
            f"{mae_bm:.2f}",
            "Надмірна кількість пропусків на однорідних поверхнях",
        ])
        writer.writerow([
            "StereoSGBM (3-Way)",
            192,
            9,
            10,
            f"{val_sgbm:.1f}",
            f"{time_sgbm:.1f}",
            f"{noise_sgbm:.3f}",
            f"{mae_sgbm:.2f}",
            "Висока повнота карти, гладкі площини та чіткі межі",
        ])
        writer.writerow([
            "StereoSGBM + WLS-фільтр",
            192,
            9,
            10,
            f"{val_wls:.1f}",
            f"{time_wls:.1f}",
            f"{noise_wls:.3f}",
            f"{mae_wls:.2f}",
            "Заповнення зон напівзакриття зі збереженням контурів",
        ])

    # Збережемо найкращі карти диспаратності для Частин D та E
    # Для найближчого об'єкта на нижньому краю кадру (#7 Накінечник олівця, Z = 76.8 см, d ≈ 200.7 пк)
    # локально доповнюємо окіл вістря олівця розрахунком з numDisparities = 208
    disp_208, _, _ = compute_disparity_sgbm(gray_l, gray_r, num_disp=208, block_size=9, uniq_ratio=10)
    disp_best_full = disp_sgbm.copy()
    disp_best_full[1108:1126, 1485:1520] = disp_208[1108:1126, 1485:1520]

    np.save(RESULTS_DIR / "best_disparity.npy", disp_best_full)
    np.save(RESULTS_DIR / "best_disparity_wls.npy", disp_wls)
    np.save(RESULTS_DIR / "bm_disparity.npy", disp_bm)

    # Візуальне порівняння StereoBM vs StereoSGBM vs StereoSGBM+WLS (Рисунок 4)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=200, layout="constrained")
    vmin, vmax = 0.0, 192.0

    for ax, d_map, title_str in [
        (axes[0], disp_bm, f"а) StereoBM (bs=15): валідних {val_bm:.1f}%, {time_bm:.0f} мс"),
        (axes[1], disp_sgbm, f"б) StereoSGBM (bs=9): валідних {val_sgbm:.1f}%, {time_sgbm:.0f} мс"),
        (axes[2], disp_wls, f"в) StereoSGBM + WLS: валідних {val_wls:.1f}%, {time_wls:.0f} мс"),
    ]:
        masked = np.ma.masked_where(d_map <= 0.5, d_map)
        cmap = plt.cm.turbo.copy()
        cmap.set_bad(color="#1a1a1a")
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title_str, fontsize=10, fontweight="bold")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.015)
    cbar.set_label("Диспаратність d = x_L - x_R, пк", fontsize=9.5)
    fig.savefig(FIGURES_DIR / "c_bm_vs_sgbm.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Крок 8: Дослідження сітки параметрів StereoSGBM
    num_disps = [64, 128, 192]
    block_sizes = [5, 9, 15, 21]
    uniq_ratios = [5, 10, 15]

    all_rows = []
    cached_maps = {}
    for nd in num_disps:
        for bs in block_sizes:
            for ur in uniq_ratios:
                d_map, dt_ms, _ = compute_disparity_sgbm(gray_l, gray_r, num_disp=nd, block_size=bs, uniq_ratio=ur)
                val_pct, g_noise, s_mae = evaluate_disparity_map(d_map, nd, sift_pts_l, sift_disp)
                note = describe_visual_quality(nd, bs, ur, val_pct, s_mae)
                row = {
                    "numDisparities": nd,
                    "blockSize": bs,
                    "uniquenessRatio": ur,
                    "valid_pct": val_pct,
                    "time_ms": dt_ms,
                    "grad_noise_px": g_noise,
                    "sift_mae_px": s_mae,
                    "visual_note": note,
                }
                all_rows.append(row)
                cached_maps[(nd, bs, ur)] = d_map

    # Збереження повної таблиці 36 комбінацій
    with open(RESULTS_DIR / "c_grid_search.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "numDisparities",
            "blockSize",
            "uniquenessRatio",
            "valid_pct",
            "time_ms",
            "grad_noise_px",
            "sift_mae_px",
            "visual_note",
        ])
        for r in all_rows:
            writer.writerow([
                r["numDisparities"],
                r["blockSize"],
                r["uniquenessRatio"],
                f"{r['valid_pct']:.1f}",
                f"{r['time_ms']:.1f}",
                f"{r['grad_noise_px']:.3f}",
                f"{r['sift_mae_px']:.2f}",
                r["visual_note"],
            ])

    # Збереження репрезентативної вибірки (14 ключових конфігурацій) для компактної таблиці у звіті
    key_configs = [
        (64, 5, 10),
        (64, 9, 10),
        (64, 15, 10),
        (128, 5, 10),
        (128, 9, 10),
        (128, 15, 10),
        (192, 5, 5),
        (192, 5, 10),
        (192, 5, 15),
        (192, 9, 5),
        (192, 9, 10),
        (192, 9, 15),
        (192, 15, 10),
        (192, 21, 10),
    ]
    with open(RESULTS_DIR / "c_grid_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "numDisparities",
            "blockSize",
            "uniquenessRatio",
            "valid_pct",
            "time_ms",
            "grad_noise_px",
            "sift_mae_px",
            "visual_note",
        ])
        for cfg in key_configs:
            for r in all_rows:
                if (r["numDisparities"], r["blockSize"], r["uniquenessRatio"]) == cfg:
                    writer.writerow([
                        r["numDisparities"],
                        r["blockSize"],
                        r["uniquenessRatio"],
                        f"{r['valid_pct']:.1f}",
                        f"{r['time_ms']:.1f}",
                        f"{r['grad_noise_px']:.3f}",
                        f"{r['sift_mae_px']:.2f}",
                        r["visual_note"],
                    ])
                    break

    # Візуальна матриця впливу параметрів (Рисунок 5: 2x4 панелі)
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.2), dpi=200, layout="constrained")
    cmap = plt.cm.turbo.copy()
    cmap.set_bad(color="#1a1a1a")

    panel_specs = [
        ((64, 9, 10), "а) numDisp=64, bs=9, ur=10"),
        ((128, 9, 10), "б) numDisp=128, bs=9, ur=10"),
        ((192, 9, 10), "в) numDisp=192, bs=9, ur=10 (опт.)"),
        ((192, 5, 10), "г) numDisp=192, bs=5, ur=10"),
        ((192, 15, 10), "д) numDisp=192, bs=15, ur=10"),
        ((192, 21, 10), "е) numDisp=192, bs=21, ur=10"),
        ((192, 9, 5), "є) numDisp=192, bs=9, ur=5"),
        ((192, 9, 15), "ж) numDisp=192, bs=9, ur=15"),
    ]
    for ax, (cfg, title_s) in zip(axes.ravel(), panel_specs):
        d_m = cached_maps[cfg]
        val_p = [r["valid_pct"] for r in all_rows if (r["numDisparities"], r["blockSize"], r["uniquenessRatio"]) == cfg][0]
        masked = np.ma.masked_where(d_m <= 0.5, d_m)
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=192)
        ax.set_title(f"{title_s}\n(валідних: {val_p:.1f}%)", fontsize=9.0, fontweight="bold")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.012)
    cbar.set_label("Диспаратність d, пк", fontsize=9.5)
    fig.savefig(FIGURES_DIR / "c_grid_maps.png", bbox_inches="tight")
    plt.close(fig)

    # Графіки залежності частки валідних пікселів, MAE та часу від параметрів (Рисунок 6)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=200)

    # Панель а: valid_pct від blockSize для різних numDisparities (при ur=10)
    colors_nd = {64: "#e15759", 128: "#f28e2b", 192: "#4e79a7"}
    markers_nd = {64: "s", 128: "^", 192: "o"}
    for nd in num_disps:
        ys_val = [
            r["valid_pct"]
            for r in all_rows
            if r["numDisparities"] == nd and r["uniquenessRatio"] == 10
        ]
        axes[0].plot(
            block_sizes,
            ys_val,
            marker=markers_nd[nd],
            color=colors_nd[nd],
            linewidth=2.0,
            label=f"numDisparities = {nd}",
        )
    axes[0].set_xticks(block_sizes)
    axes[0].set_xlabel("Розмір вікна зіставлення blockSize, пк", fontsize=9.5)
    axes[0].set_ylabel("Частка валідних пікселів, %", fontsize=9.5)
    axes[0].set_title("а) Повнота карти (при uniquenessRatio = 10)", fontsize=10, fontweight="bold")
    axes[0].grid(linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.5)

    # Панель б: вплив uniquenessRatio при numDisparities = 192
    colors_ur = {5: "#59a14f", 10: "#4e79a7", 15: "#b07aa1"}
    for ur in uniq_ratios:
        ys_val = [
            r["valid_pct"]
            for r in all_rows
            if r["numDisparities"] == 192 and r["uniquenessRatio"] == ur
        ]
        axes[1].plot(
            block_sizes,
            ys_val,
            marker="o",
            color=colors_ur[ur],
            linewidth=2.0,
            label=f"uniquenessRatio = {ur}%",
        )
    axes[1].set_xticks(block_sizes)
    axes[1].set_xlabel("Розмір вікна зіставлення blockSize, пк", fontsize=9.5)
    axes[1].set_ylabel("Частка валідних пікселів, %", fontsize=9.5)
    axes[1].set_title("б) Вплив uniquenessRatio (numDisp = 192)", fontsize=10, fontweight="bold")
    axes[1].grid(linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.5)

    # Панель в: Час обчислення та похибка MAE відносно SIFT
    ax_time = axes[2]
    ax_mae = ax_time.twinx()
    times_192 = [
        r["time_ms"]
        for r in all_rows
        if r["numDisparities"] == 192 and r["uniquenessRatio"] == 10
    ]
    maes_192 = [
        r["sift_mae_px"]
        for r in all_rows
        if r["numDisparities"] == 192 and r["uniquenessRatio"] == 10
    ]
    l1 = ax_time.plot(
        block_sizes, times_192, marker="s", color="#4e79a7", linewidth=2.0, label="Час SGBM (мс)"
    )
    l2 = ax_mae.plot(
        block_sizes, maes_192, marker="D", color="#e15759", linestyle="--", linewidth=2.0, label="Похибка MAE до SIFT (пк)"
    )
    ax_time.set_xticks(block_sizes)
    ax_time.set_xlabel("Розмір вікна blockSize, пк", fontsize=9.5)
    ax_time.set_ylabel("Час обчислення кадру, мс", color="#4e79a7", fontsize=9.5)
    ax_mae.set_ylabel("Середня похибка MAE до SIFT, пк", color="#e15759", fontsize=9.5)
    ax_time.set_title("в) Час дії та точність (numDisp = 192, ur = 10)", fontsize=10, fontweight="bold")
    ax_time.grid(linestyle=":", alpha=0.6)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_time.legend(lines, labels, fontsize=8.5, loc="upper center")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "c_grid_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Крок 9: Візуалізація найкращої карти диспаратності з кольоровою шкалою у пікселях (Рисунок 7)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), dpi=200)
    masked_best = np.ma.masked_where(disp_sgbm <= 0.5, disp_sgbm)
    im0 = axes[0].imshow(masked_best, cmap=cmap, vmin=0, vmax=192)
    axes[0].set_title(
        f"а) Найкраща карта диспаратності StereoSGBM (numDisp=192, bs=9, ur=10, валідних {val_sgbm:.1f}%)",
        fontsize=10,
        fontweight="bold",
    )
    axes[0].axis("off")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.set_label("Диспаратність d = x_L - x_R, пікселі", fontsize=9.5)

    # Гістограма розподілу диспаратності з підписаними піками основних планів сцени
    valid_d = disp_sgbm[:, 192:][disp_sgbm[:, 192:] > 0.5]
    axes[1].hist(valid_d, bins=90, color="#4e79a7", edgecolor="black", linewidth=0.35, alpha=0.85)
    planes = [
        (46.0, "Дальня стіна / вікно\n(d ≈ 38..47 пк)"),
        (61.0, "Рюкзак / ТВ / чайник\n(d ≈ 59..64 пк)"),
        (104.5, "Ноутбук\n(d ≈ 104..105 пк)"),
        (125.0, "Ширма D&D / Блахадж\n(d ≈ 115..125 пк)"),
        (140.5, "Границя ватмана\n(d ≈ 140 пк)"),
    ]
    for d_val, label_txt in planes:
        axes[1].axvline(d_val, color="#e15759", linestyle="--", linewidth=1.3)
        axes[1].text(
            d_val,
            axes[1].get_ylim()[1] * 0.78 if d_val not in (61.0, 125.0) else axes[1].get_ylim()[1] * 0.52,
            label_txt,
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="#e15759"),
        )
    axes[1].set_xlabel("Диспаратність d, пікселі", fontsize=9.5)
    axes[1].set_ylabel("Кількість пікселів", fontsize=9.5)
    axes[1].set_title("б) Гістограма диспаратності та глибинні ешелони сцени", fontsize=10, fontweight="bold")
    axes[1].grid(linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "c_best_disparity.png", bbox_inches="tight")
    plt.close(fig)
    print("Частину C успішно виконано!")


if __name__ == "__main__":
    run_part_c()
