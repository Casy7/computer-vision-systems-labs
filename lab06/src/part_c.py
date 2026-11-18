"""
Лабораторна робота № 6.
Частина C: Щільний оптичний потік (Farneback) та порівняння з розрідженим (Кроки 8–10).
"""

import gc
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    RESULTS_DIR, FIGURES_DIR,
    VIDEO_A_PATH, VIDEO_B_PATH,
    save_csv, load_video_frames, build_clean_background, active_roi_mask,
)


def flow_to_hsv_bgr(flow):
    """
    Крок 9: Перетворення щільного поля векторів (u, v) у HSV -> BGR:
      H = кут (ang / 2), S = 255, V = нормалізована величина потоку.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(ang / 2.0, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr, mag, ang


def make_hsv_color_wheel(size=220):
    """Колірне колесо напрямків HSV для легенди щільного потоку."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2.0, size / 2.0
    dx = xx - cx
    dy = yy - cy
    mag, ang = cv2.cartToPolar(dx.astype(np.float32), dy.astype(np.float32), angleInDegrees=True)
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    r_max = size * 0.48
    inside = mag <= r_max
    hsv[..., 0] = np.clip(ang / 2.0, 0, 179).astype(np.uint8)
    hsv[..., 1] = np.where(inside, 255, 0).astype(np.uint8)
    hsv[..., 2] = np.where(inside, np.clip(mag / r_max * 255.0, 0, 255), 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def draw_quiver_overlay(frame_bgr, flow, step=45, min_mag=1.2):
    """Малює розріджену сітку векторів щільного потоку поверх кадру 2592x1944."""
    vis = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            mag = float(np.hypot(fx, fy))
            if mag >= min_mag:
                p1 = (int(x), int(y))
                p2 = (int(round(x + fx * 3.0)), int(round(y + fy * 3.0)))
                cv2.arrowedLine(vis, p1, p2, (0, 255, 180), 3, cv2.LINE_AA, tipLength=0.35)
    return vis


def run_part_c():
    print("=" * 65)
    print("Частина C: Щільний оптичний потік Farneback та порівняння (Кроки 8–10)")
    print("=" * 65)

    farneback_params = dict(
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    motion_rows = []
    saved_panels = []

    def analyze_pairs(vname, frames, bg, spec_pairs):
        h, w = frames[0].shape[:2]
        for idx, desc in spec_pairs:
            f1 = frames[idx]
            f2 = frames[idx + 1]
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

            t0 = time.perf_counter()
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, **farneback_params)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            hsv_bgr, mag, _ = flow_to_hsv_bgr(flow)
            quiver_bgr = draw_quiver_overlay(f1, flow, step=45, min_mag=1.2)

            diff = np.abs(f1.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
            obj_mask = diff > 28
            bg_mask = ~obj_mask

            mag_obj = mag[obj_mask] if obj_mask.any() else np.array([0.0])
            mag_bg = mag[bg_mask] if bg_mask.any() else np.array([0.0])

            motion_rows.append([
                vname, idx, desc,
                f"{dt_ms:.1f}",
                f"{mag.mean():.3f}",
                f"{mag.max():.2f}",
                f"{mag_obj.mean():.2f}",
                f"{np.percentile(mag_obj, 95):.2f}",
                f"{mag_bg.mean():.3f}",
            ])
            print(f"  {vname} (кадр {idx}->{idx+1}): час={dt_ms:.1f} мс, "
                  f"об'єкт mean={mag_obj.mean():.2f} пк (p95={np.percentile(mag_obj, 95):.2f} пк), "
                  f"фон mean={mag_bg.mean():.3f} пк, max={mag.max():.2f} пк")
            saved_panels.append((vname, idx, desc, quiver_bgr, hsv_bgr, float(mag.max())))

    # 1. Обробка Відео A та бенчмарк швидкодії на 2592x1944
    frames_a, fps_a, w, h = load_video_frames(VIDEO_A_PATH)
    bg_a = build_clean_background(frames_a, mode="a")
    analyze_pairs("Відео A", frames_a, bg_a, [
        (160, "Акула рухається вгору-ліворуч"),
        (490, "Акула прямує вгору-ліворуч після розвороту"),
    ])

    print(f"\nБенчмарк швидкодії на Відео A ({w}x{h})...")
    n_bench = 30
    n_bench_full = 15
    lk_params = dict(
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    grays_a = [cv2.cvtColor(frames_a[i], cv2.COLOR_BGR2GRAY) for i in range(n_bench + 1)]
    grays_a_q = [cv2.resize(g, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) for g in grays_a]
    grays_a_h = [cv2.resize(g, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for g in grays_a]

    p0 = cv2.goodFeaturesToTrack(grays_a[0], maxCorners=300, qualityLevel=0.01, minDistance=10, mask=active_roi_mask(h, w, 10))

    # 1) PyrLK (прямий прохід, 2592x1944)
    pts = p0.copy()
    times_lk_fwd = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        p1, _, _ = cv2.calcOpticalFlowPyrLK(grays_a[i], grays_a[i + 1], pts, None, **lk_params)
        times_lk_fwd.append((time.perf_counter() - t0) * 1000.0)
        pts = p1

    # 2) PyrLK + зворотна перевірка (2592x1944)
    pts = p0.copy()
    times_lk_fb = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        p1, _, _ = cv2.calcOpticalFlowPyrLK(grays_a[i], grays_a[i + 1], pts, None, **lk_params)
        _, _, _ = cv2.calcOpticalFlowPyrLK(grays_a[i + 1], grays_a[i], p1, None, **lk_params)
        times_lk_fb.append((time.perf_counter() - t0) * 1000.0)
        pts = p1

    # 3) Farneback (масштаб 0.25 -> 648x486)
    times_fb_q = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        _ = cv2.calcOpticalFlowFarneback(grays_a_q[i], grays_a_q[i + 1], None, **farneback_params)
        times_fb_q.append((time.perf_counter() - t0) * 1000.0)

    # 4) Farneback (масштаб 0.50 -> 1296x972)
    times_fb_h = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        _ = cv2.calcOpticalFlowFarneback(grays_a_h[i], grays_a_h[i + 1], None, **farneback_params)
        times_fb_h.append((time.perf_counter() - t0) * 1000.0)

    # 5) Farneback (повний кадр 2592x1944, 15 пар)
    times_fb_full = []
    for i in range(n_bench_full):
        t0 = time.perf_counter()
        _ = cv2.calcOpticalFlowFarneback(grays_a[i], grays_a[i + 1], None, **farneback_params)
        times_fb_full.append((time.perf_counter() - t0) * 1000.0)

    del frames_a, bg_a, grays_a, grays_a_q, grays_a_h
    gc.collect()

    # 2. Обробка Відео B
    frames_b, _, _, _ = load_video_frames(VIDEO_B_PATH)
    bg_b = build_clean_background(frames_b, mode="b")
    analyze_pairs("Відео B", frames_b, bg_b, [
        (30, "Машинка їде праворуч-вгору, акула праворуч"),
        (310, "Машинка виїжджає з-під перекриття (швидкий рух)"),
    ])
    del frames_b, bg_b
    gc.collect()

    save_csv(
        RESULTS_DIR / "c_flow_motion_stats.csv",
        ["video_id", "frame_idx", "episode", "time_ms", "mean_mag_frame", "max_mag_px", "obj_mean_mag", "obj_p95_mag", "bg_mean_mag"],
        motion_rows
    )

    # Окремі HSV-кадри (Крок 9)
    cv2.imwrite(str(RESULTS_DIR / "c_farneback_hsv_video_a.png"), saved_panels[1][4])
    cv2.imwrite(str(RESULTS_DIR / "c_farneback_hsv_video_b.png"), saved_panels[3][4])

    # Композитна ілюстрація c_dense_flow_hsv.png
    fig = plt.figure(figsize=(11.6, 7.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.38])

    v_a = saved_panels[1]
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(cv2.cvtColor(v_a[3], cv2.COLOR_BGR2RGB))
    ax00.set_title(f"(а) {v_a[0]}, кадр {v_a[1]}: кадри + вектори потоку", fontsize=9.5)
    ax00.axis("off")

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(cv2.cvtColor(v_a[4], cv2.COLOR_BGR2RGB))
    ax01.set_title(f"(б) HSV-поле {v_a[0]} ($|\\mathbf{{v}}|_\\max = {v_a[5]:.1f}$ пк/кадр)", fontsize=9.5)
    ax01.axis("off")

    v_b = saved_panels[3]
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(cv2.cvtColor(v_b[3], cv2.COLOR_BGR2RGB))
    ax10.set_title(f"(в) {v_b[0]}, кадр {v_b[1]}: вихід з-під перекриття", fontsize=9.5)
    ax10.axis("off")

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.imshow(cv2.cvtColor(v_b[4], cv2.COLOR_BGR2RGB))
    ax11.set_title(f"(г) HSV-поле {v_b[0]} ($|\\mathbf{{v}}|_\\max = {v_b[5]:.1f}$ пк/кадр)", fontsize=9.5)
    ax11.axis("off")

    ax_wheel = fig.add_subplot(gs[:, 2])
    wheel = make_hsv_color_wheel(220)
    ax_wheel.imshow(wheel)
    ax_wheel.set_title("Колірна шкала\nнапрямку $H$ (HSV)", fontsize=9.5, pad=8)
    ax_wheel.text(222, 110, "0°", va="center", ha="left", fontsize=8.5, fontweight="bold")
    ax_wheel.text(110, -4, "270° (вгору)", va="bottom", ha="center", fontsize=8.2, fontweight="bold")
    ax_wheel.text(-4, 110, "180°", va="center", ha="right", fontsize=8.5, fontweight="bold")
    ax_wheel.text(110, 224, "90° (вниз)", va="top", ha="center", fontsize=8.2, fontweight="bold")
    ax_wheel.set_xlim(-48, 255)
    ax_wheel.set_ylim(252, -32)
    ax_wheel.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "c_dense_flow_hsv.png")
    plt.close(fig)

    bench_specs = [
        ("Розріджений PyrLK (прямий прохід)", f"{w}x{h}", f"{len(p0)} точок", np.array(times_lk_fwd), "Відстеження ознак, SLAM/одом."),
        ("Розріджений PyrLK + FB-перевірка", f"{w}x{h}", f"{len(p0)} точок", np.array(times_lk_fb), "Стійкі траєкторії точок"),
        ("Щільний Farneback (масштаб 0.25)", "648x486", "314 928 пікселів", np.array(times_fb_q), "Реалтайм-оцінка поля руху"),
        ("Щільний Farneback (масштаб 0.50)", "1296x972", "1 259 712 пікселів", np.array(times_fb_h), "Баланс точності та швидкості"),
        ("Щільний Farneback (повний кадр)", f"{w}x{h}", "5 038 848 пікселів", np.array(times_fb_full), "Попіксельний вектор руху"),
    ]

    comp_rows = []
    for m_name, res_str, elems, arr, suit in bench_specs:
        mean_ms = float(arr.mean())
        std_ms = float(arr.std())
        fps_eq = 1000.0 / mean_ms
        comp_rows.append([m_name, res_str, elems, f"{mean_ms:.2f}", f"{std_ms:.2f}", f"{fps_eq:.1f}", suit])
        print(f"  {m_name:36s} ({res_str}): {mean_ms:7.2f} +- {std_ms:5.2f} мс/кадр ({fps_eq:6.1f} FPS)")

    save_csv(
        RESULTS_DIR / "c_sparse_vs_dense.csv",
        ["method", "resolution", "elements", "mean_ms", "std_ms", "fps_equivalent", "best_for"],
        comp_rows
    )

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    labels_short = [
        "PyrLK\n(прямий)",
        "PyrLK +\nFB-перевірка",
        "Farneback\n(648×486)",
        "Farneback\n(1296×972)",
        "Farneback\n(2592×1944)",
    ]
    means = [float(x[3]) for x in comp_rows]
    stds = [float(x[4]) for x in comp_rows]
    fps_vals = [float(x[5]) for x in comp_rows]
    colors = ["#1b9e77", "#66a61e", "#7570b3", "#e7298a", "#d95f02"]

    x_pos = np.arange(len(labels_short))
    bars1 = ax_t.bar(x_pos, means, yerr=stds, capsize=4, color=colors, alpha=0.88, edgecolor="black", lw=0.6)
    ax_t.set_yscale("log")
    ax_t.axhline(1000.0 / fps_a, color="#b2182b", ls="--", lw=1.4, label=f"Поріг 30 FPS ({1000.0/fps_a:.1f} мс)")
    ax_t.set_xticks(x_pos)
    ax_t.set_xticklabels(labels_short, fontsize=8.5)
    ax_t.set_ylabel("Час обчислення на кадр, мс (лог. шкала)")
    ax_t.set_title("(а) Витрати часу на одну пару кадрів")
    ax_t.set_ylim(min(means) * 0.5, max(means) * 2.5)
    ax_t.grid(True, which="both", axis="y", ls=":", alpha=0.6)
    ax_t.legend(loc="upper left", fontsize=8.5)
    for b, m in zip(bars1, means):
        ax_t.text(b.get_x() + b.get_width() / 2.0, m * 1.16, f"{m:.1f} мс", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    bars2 = ax_f.bar(x_pos, fps_vals, color=colors, alpha=0.88, edgecolor="black", lw=0.6)
    ax_f.set_yscale("log")
    ax_f.axhline(fps_a, color="#b2182b", ls="--", lw=1.4, label=f"Частота відео ({fps_a:.0f} кадр/с)")
    ax_f.set_xticks(x_pos)
    ax_f.set_xticklabels(labels_short, fontsize=8.5)
    ax_f.set_ylabel("Еквівалентна швидкодія, кадр/с (лог. шкала)")
    ax_f.set_title("(б) Пропускна здатність алгоритмів (FPS)")
    ax_f.set_ylim(0.5, max(fps_vals) * 2.6)
    ax_f.grid(True, which="both", axis="y", ls=":", alpha=0.6)
    ax_f.legend(loc="upper right", fontsize=8.5)
    for b, fv in zip(bars2, fps_vals):
        ax_f.text(b.get_x() + b.get_width() / 2.0, fv * 1.16, f"{fv:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "c_flow_comparison.png")
    plt.close(fig)

    print("Частину C успішно виконано!")


if __name__ == "__main__":
    run_part_c()
