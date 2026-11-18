"""
Лабораторна робота № 6.
Частина D: Віднімання фону MOG2, морфологічна постобробка та підрахунок об'єктів
через контрольну лінію (Кроки 11–15).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    RESULTS_DIR, FIGURES_DIR,
    VIDEO_A_PATH,
    save_csv, load_video_frames, build_clean_background, make_video_writer,
)


def run_mog2_pipeline(frames, bg_a, warm_start=True, lr=0.0005, min_area=4500):
    """
    Виконання кроків 11–13:
      MOG2(history=500, varThreshold=16, detectShadows=True) -> відсікання тіней (127) ->
      морфологічне відкриття 3x3 + закриття 5x5 -> connectedComponentsWithStats з порогом площі.
    """
    mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    if warm_start:
        for _ in range(30):
            mog.apply(bg_a, learningRate=0.1)

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    stats_raw_px, stats_shad_px, stats_nosh_px, stats_morph_px, stats_final_px = [], [], [], [], []
    comp_nosh, comp_morph, comp_final = [], [], []
    centroids_y = []
    centroids_x = []
    bboxes = []
    snapshots = {}

    for f_idx, fr in enumerate(frames):
        raw = mog.apply(fr, learningRate=lr)

        shad_mask = (raw == 127)
        fg_255 = (raw == 255)
        stats_raw_px.append(int((fg_255 | shad_mask).sum()))
        stats_shad_px.append(int(shad_mask.sum()))
        stats_nosh_px.append(int(fg_255.sum()))

        # Крок 12: видалення тіней (127) порогуванням (> 200 залишає лише 255)
        _, no_shadow = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        n_ns, _, _, _ = cv2.connectedComponentsWithStats(no_shadow)
        comp_nosh.append(n_ns - 1)

        # Крок 13: відкриття 3x3, потім закриття 5x5
        opened = cv2.morphologyEx(no_shadow, cv2.MORPH_OPEN, k_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
        stats_morph_px.append(int((closed > 0).sum()))

        num_c, labels, stats_c, cent_c = cv2.connectedComponentsWithStats(closed)
        comp_morph.append(num_c - 1)

        valid_ids = [i for i in range(1, num_c) if stats_c[i, cv2.CC_STAT_AREA] >= min_area]
        comp_final.append(len(valid_ids))

        if valid_ids:
            best = max(valid_ids, key=lambda i: stats_c[i, cv2.CC_STAT_AREA])
            final_mask = np.where(labels == best, 255, 0).astype(np.uint8)
            stats_final_px.append(int(stats_c[best, cv2.CC_STAT_AREA]))
            centroids_x.append(float(cent_c[best, 0]))
            centroids_y.append(float(cent_c[best, 1]))
            bboxes.append(tuple(int(v) for v in stats_c[best, :4]))
        else:
            final_mask = np.zeros_like(closed)
            stats_final_px.append(0)
            centroids_x.append(np.nan)
            centroids_y.append(np.nan)
            bboxes.append(None)

        if f_idx == 450:
            snapshots[f_idx] = (fr.copy(), raw.copy(), no_shadow.copy(), closed.copy(), final_mask.copy())

    return {
        "raw_px": np.array(stats_raw_px),
        "shad_px": np.array(stats_shad_px),
        "nosh_px": np.array(stats_nosh_px),
        "morph_px": np.array(stats_morph_px),
        "final_px": np.array(stats_final_px),
        "comp_nosh": np.array(comp_nosh),
        "comp_morph": np.array(comp_morph),
        "comp_final": np.array(comp_final),
        "cx": np.array(centroids_x),
        "cy": np.array(centroids_y),
        "bboxes": bboxes,
        "snapshots": snapshots,
    }


def count_line_crossings(cy_arr, y_line, deadband=16.0):
    """
    Крок 14: Підрахунок перетинів горизонтальної лінії y = y_line.
    Повертає результати як наївного алгоритму (без гістерезису), так і захищеного
    від подвійного підрахунку алгоритму зі скінченним автоматом та мертвою зоною +-deadband.
    """
    naive_events = []
    valid_idx = np.where(np.isfinite(cy_arr))[0]
    for k in range(1, len(valid_idx)):
        i_prev, i_cur = valid_idx[k - 1], valid_idx[k]
        d_prev = cy_arr[i_prev] - y_line
        d_cur = cy_arr[i_cur] - y_line
        if (d_prev <= 0 < d_cur) or (d_prev >= 0 > d_cur):
            direction = "зверху вниз" if d_cur > 0 else "знизу вгору"
            naive_events.append((int(i_cur), direction, float(cy_arr[i_cur])))

    deb_events = []
    if len(valid_idx) > 0:
        state = 1 if cy_arr[valid_idx[0]] > y_line else -1
        for i in valid_idx:
            yv = cy_arr[i]
            if state == 1 and yv < y_line - deadband:
                state = -1
                deb_events.append((int(i), "знизу вгору", float(yv)))
            elif state == -1 and yv > y_line + deadband:
                state = 1
                deb_events.append((int(i), "зверху вниз", float(yv)))

    return naive_events, deb_events


def run_part_d():
    print("=" * 65)
    print("Частина D: Віднімання фону MOG2 і підрахунок через лінію (Кроки 11–15)")
    print("=" * 65)

    frames_a, fps_a, w, h = load_video_frames(VIDEO_A_PATH)
    bg_a = build_clean_background(frames_a, mode="a")

    # Еталонна перевірка траєкторії центроїда по чистому фону (для звірки Кроку 15)
    gt_y = []
    for fr in frames_a:
        diff = np.abs(fr.astype(np.int16) - bg_a.astype(np.int16)).max(axis=2)
        m = cv2.morphologyEx((diff > 30).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        num, _, st, cent = cv2.connectedComponentsWithStats(m)
        if num > 1:
            best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            gt_y.append(float(cent[best, 1]))
        else:
            gt_y.append(np.nan)
    gt_y = np.array(gt_y)

    # Запуск MOG2: ініціалізований фон і холодний старт
    res_warm = run_mog2_pipeline(frames_a, bg_a, warm_start=True, lr=0.0005, min_area=4500)
    res_cold = run_mog2_pipeline(frames_a, bg_a, warm_start=False, lr=-1.0, min_area=4500)

    det_warm = int(np.isfinite(res_warm["cy"]).sum())
    det_cold = int(np.isfinite(res_cold["cy"]).sum())
    print(f"Виявлено об'єкт: ініціалізований MOG2 = {det_warm}/{len(frames_a)} кадрів, холодний старт = {det_cold}/{len(frames_a)}")

    mean_raw = float(res_warm["raw_px"].mean())
    stage_rows = [
        ["1. Сира маска MOG2 (об'єкт + тіні 127)", f"{mean_raw:.0f}", f"{res_warm['shad_px'].mean():.0f}", "100.0", "—", f"{det_warm}/{len(frames_a)}"],
        ["2. Порогування тіней (raw > 200)", f"{res_warm['nosh_px'].mean():.0f}", "0", f"{res_warm['nosh_px'].mean()/mean_raw*100:.1f}", f"{res_warm['comp_nosh'].mean():.1f}", f"{det_warm}/{len(frames_a)}"],
        ["3. Морфологія (відкр. 3x3 + закр. 5x5)", f"{res_warm['morph_px'].mean():.0f}", "0", f"{res_warm['morph_px'].mean()/mean_raw*100:.1f}", f"{res_warm['comp_morph'].mean():.1f}", f"{det_warm}/{len(frames_a)}"],
        ["4. Фільтр площі (S >= 4500 пк)", f"{res_warm['final_px'].mean():.0f}", "0", f"{res_warm['final_px'].mean()/mean_raw*100:.1f}", f"{res_warm['comp_final'].mean():.2f}", f"{det_warm}/{len(frames_a)}"],
    ]
    save_csv(
        RESULTS_DIR / "d_mog2_stages.csv",
        ["stage", "mean_fg_px", "mean_shadow_px", "pct_of_raw", "mean_components", "detected_frames"],
        stage_rows
    )

    # Рисунок етапів постобробки MOG2 на кадрі 450
    snap_f, raw_m, nosh_m, _, _ = res_warm["snapshots"][450]
    raw_vis = np.zeros((*raw_m.shape, 3), dtype=np.uint8)
    raw_vis[raw_m == 127] = (120, 120, 120)
    raw_vis[raw_m == 255] = (255, 255, 255)
    det_vis = snap_f.copy()
    bx, by, bw, bh = res_warm["bboxes"][450]
    cx_450, cy_450 = int(round(res_warm["cx"][450])), int(round(res_warm["cy"][450]))
    cv2.rectangle(det_vis, (bx, by), (bx + bw, by + bh), (0, 220, 0), 5)
    cv2.circle(det_vis, (cx_450, cy_450), 14, (0, 0, 255), -1)

    fig, axs = plt.subplots(2, 2, figsize=(11.2, 7.2))
    axs[0, 0].imshow(cv2.cvtColor(snap_f, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("(а) Вхідний кадр № 450 Відео A (акула на світлому паркеті)", fontsize=9.5)
    axs[0, 0].axis("off")

    axs[0, 1].imshow(raw_vis)
    axs[0, 1].set_title(f"(б) Сира маска MOG2: білий — об'єкт, сірий (127) — тінь ({int((raw_m==127).sum())} пк)", fontsize=9.5)
    axs[0, 1].axis("off")

    axs[1, 0].imshow(nosh_m, cmap="gray")
    axs[1, 0].set_title(f"(в) Після видалення тіней порогуванням (дрібний шум: {int(res_warm['comp_nosh'][450])} комп.)", fontsize=9.5)
    axs[1, 0].axis("off")

    axs[1, 1].imshow(cv2.cvtColor(det_vis, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title(f"(г) Морфологія (3×3, 5×5) + фільтр площі: S={int(res_warm['final_px'][450])} пк", fontsize=9.5)
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "d_mog2_stages.png")
    plt.close(fig)

    # Кроки 14–15: Підрахунок перетинів контрольної горизонтальної лінії
    y_line_main = 1116
    line_rows = []
    event_rows = []
    main_naive, main_deb = None, None
    for y_l in (1100, 1116, 1152):
        _, gt_ev = count_line_crossings(gt_y, y_l, deadband=16.0)
        c_naive, _ = count_line_crossings(res_cold["cy"], y_l, deadband=16.0)
        w_naive, w_deb = count_line_crossings(res_warm["cy"], y_l, deadband=16.0)
        gt_cnt = len(gt_ev)
        up_cnt = sum(1 for e in w_deb if "вгору" in e[1])
        dn_cnt = sum(1 for e in w_deb if "вниз" in e[1])
        acc = (min(len(w_deb), gt_cnt) / gt_cnt * 100.0) if len(w_deb) == gt_cnt else (max(0.0, 1.0 - abs(len(w_deb) - gt_cnt) / gt_cnt) * 100.0)
        line_rows.append([y_l, gt_cnt, len(c_naive), len(w_naive), len(w_deb), up_cnt, dn_cnt, f"{acc:.1f}"])
        print(f"  Лінія y={y_l}: еталон={gt_cnt}, холодний старт MOG2={len(c_naive)}, "
              f"без гістерезису={len(w_naive)}, із гістерезисом (+-16 пк)={len(w_deb)} "
              f"(вгору={up_cnt}, вниз={dn_cnt})")
        if y_l == y_line_main:
            main_naive, main_deb = w_naive, w_deb
            for k_ev, (fr_i, dir_s, yv) in enumerate(w_deb, start=1):
                event_rows.append([k_ev, fr_i, f"{fr_i / fps_a:.2f}", dir_s, f"{yv:.1f}", y_l])

    save_csv(
        RESULTS_DIR / "d_line_crossing.csv",
        ["y_line_px", "manual_gt", "cold_mog2_naive", "warm_mog2_no_debounce", "warm_mog2_debounced", "up_count", "down_count", "accuracy_pct"],
        line_rows
    )
    save_csv(
        RESULTS_DIR / "d_crossing_events.csv",
        ["event_id", "frame_idx", "time_s", "direction", "centroid_y_px", "y_line_px"],
        event_rows
    )

    # Запис відео з підрахунком перетинів лінії (Крок 14)
    writer, _ = make_video_writer(RESULTS_DIR / "d_line_crossing_video_a.mp4", fps_a, (w, h))
    deb_frames_set = {e[0]: (idx + 1, e[1]) for idx, e in enumerate(main_deb)}
    running_count = 0
    for f_idx, fr in enumerate(frames_a):
        vis = fr.copy()
        if f_idx in deb_frames_set:
            running_count = deb_frames_set[f_idx][0]
        line_col = (0, 255, 0) if f_idx in deb_frames_set else (0, 140, 255)
        cv2.line(vis, (0, y_line_main), (w, y_line_main), line_col, 4)
        cv2.line(vis, (0, y_line_main - 16), (w, y_line_main - 16), (180, 180, 180), 2)
        cv2.line(vis, (0, y_line_main + 16), (w, y_line_main + 16), (180, 180, 180), 2)
        bb = res_warm["bboxes"][f_idx]
        if bb is not None:
            x0, y0, bw, bh = bb
            cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + bh), (0, 230, 0), 4)
            cv2.circle(vis, (int(round(res_warm["cx"][f_idx])), int(round(res_warm["cy"][f_idx]))), 12, (0, 0, 255), -1)
        cv2.rectangle(vis, (24, 24), (860, 92), (20, 20, 20), -1)
        cv2.putText(vis, f"Frame {f_idx:3d} | Line y={y_line_main} | Count: {running_count}",
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(vis)
    writer.release()

    # Графік траєкторії центроїда y_c(t) та кадри перетину (d_line_crossing.png)
    fig = plt.figure(figsize=(11.4, 7.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0])
    ax_tr = fig.add_subplot(gs[0, :])
    fr_idx_arr = np.arange(len(frames_a))
    ax_tr.plot(fr_idx_arr, res_warm["cy"], color="#1f78b4", lw=2.0, label="Координата центроїда $y_c(t)$ (MOG2 + постобробка)")
    ax_tr.axhline(y_line_main, color="#e31a1c", lw=1.6, ls="--", label=f"Контрольна лінія $y = {y_line_main}$ пк")
    ax_tr.axhspan(y_line_main - 16, y_line_main + 16, color="#fb9a99", alpha=0.28, label="Зона гістерезису $\\pm 16$ пк")

    for k_ev, (f_ev, dir_s, yv) in enumerate(main_deb, start=1):
        is_up = "вгору" in dir_s
        marker = "^" if is_up else "v"
        mcol = "#1b9e77" if is_up else "#7570b3"
        ax_tr.scatter([f_ev], [yv], color=mcol, marker=marker, s=90, zorder=5, edgecolor="black", lw=0.7)
        y_off = 14 if is_up else -18
        x_off = -14 if k_ev in (2, 4) else (14 if k_ev in (3, 5) else 0)
        ax_tr.annotate(f"#{k_ev} ({f_ev})", (f_ev, yv), textcoords="offset points", xytext=(x_off, y_off),
                       ha="center", fontsize=8.2, fontweight="bold", color=mcol)

    # Позначка зайвих спрацювань без гістерезису (якщо є)
    deb_f = [e[0] for e in main_deb]
    extra_naive = [e for e in main_naive if not any(abs(e[0] - df) <= 10 for df in deb_f)]
    if extra_naive:
        ax_tr.scatter([e[0] for e in extra_naive], [e[2] for e in extra_naive], color="#d95f02", marker="x", s=70, zorder=6,
                      label="Хибні подвійні спрацювання без гістерезису")

    ax_tr.set_ylim(1660, 740)
    ax_tr.set_xlabel("Номер кадру")
    ax_tr.set_ylabel("Вертикальна координата $y_c$, пк")
    ax_tr.set_title(f"(а) Траєкторія центроїда й підрахунок перетинів лінії $y = {y_line_main}$ пк (еталон = 8, із гістерезисом = {len(main_deb)}, без гістерезису = {len(main_naive)})")
    ax_tr.grid(True, ls=":", alpha=0.6)
    ax_tr.legend(loc="upper left", ncol=2, fontsize=8.2)

    pick_idx = [0, 1, 6, len(main_deb) - 1]
    sub_labels = ["(б) Перетин #1 (вниз)", "(в) Перетин #2 (вгору)", "(г) Перетин #7 (вниз)", f"(д) Перетин #{len(main_deb)} (вгору)"]
    for col_i, (p_i, slab) in enumerate(zip(pick_idx, sub_labels)):
        f_ev, dir_s, yv = main_deb[p_i]
        ax_sub = fig.add_subplot(gs[1, col_i])
        im = frames_a[f_ev].copy()
        bb = res_warm["bboxes"][f_ev]
        if bb is not None:
            x0, y0, bw, bh = bb
            cv2.rectangle(im, (x0, y0), (x0 + bw, y0 + bh), (0, 230, 0), 6)
        cv2.line(im, (0, y_line_main), (w, y_line_main), (0, 0, 255), 6)
        cv2.circle(im, (int(round(res_warm["cx"][f_ev])), int(round(yv))), 16, (0, 255, 255), -1)
        ax_sub.imshow(cv2.cvtColor(im[450:1760, 150:2450], cv2.COLOR_BGR2RGB))
        ax_sub.set_title(f"{slab}\nкадр {f_ev}, ліч. = {p_i + 1}", fontsize=9)
        ax_sub.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "d_line_crossing.png")
    plt.close(fig)

    print("Частину D успішно виконано!")


if __name__ == "__main__":
    run_part_d()
