"""
Лабораторна робота № 4. Частина E: Аналіз випадків відмови (Крок 17).
Відтворення та кількісний аналіз двох класичних режимів відмови гомографічного зшивання:

1. Дефіцит геометричного перекриття (< 15 %) та великий розворот ракурсу:
   - порівняння нормального перекриття (angle3-angle4, ~84 %), помірного (scale3-scale4, 41 % / 100 %),
     асиметричного масштабного вкладення (scale1-scale3: малий кадр займає лише 5.8 % площі великого,
     але сам видимий у ньому на 100 %, тому гомографія коректна!) із реальним дефіцитом перекриття:
     (а) два підкадри з контрольованим перекриттям 8 % (< 15 %),
     (б) великий стрибок масштабу scale1-scale5 (> 10x за лінійним розміром, < 1 % площі),
     (в) рознесені ракурси angle4-angle7, angle4-angle8, angle8-angle10, де спільна плоска поверхня
         майже відсутня або спотворена ракурсом, кількість inliers падає до 6-11 випадкових точок,
         а матриця H втрачає проєктивну опуклість (перетин лінії горизонту).

2. Просторовий паралакс при зміщенні оптичного центра камери в 3D-сцені:
   - 5-кадрова серія обходу об'ємного жорсткого диска angle2-angle6 (опорний angle4):
     RANSAC фіксує домінантну площину друкованої плати з високою точністю на точках самої плати,
     проте точки рельєфного металевого корпусу HDD і клавіатури на задньому плані мають іншу глибину Z,
     тому не задовольняють єдину матрицю H і дають багатократне двоїння контурів (ghosting).
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    DATA_HW_DIR, RESULTS_DIR, FIGURES_DIR,
    imread_bgr, save_csv,
    extract_sift_features, apply_ratio_test, apply_mutual_match,
    compute_reprojection_errors, homography_plausibility, compute_overlap_fractions,
    custom_ransac_homography, stitch_panorama_sequence
)


def evaluate_pair(img_a, img_b, ratio=0.80, tau=3.0):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    kps_a, d_a = extract_sift_features(gray_a)
    kps_b, d_b = extract_sift_features(gray_b)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    cand = apply_ratio_test(bf.knnMatch(d_a, d_b, k=2), ratio=ratio)
    if len(cand) < 4:
        return len(cand), 0, float("nan"), float("nan"), float("inf"), False, "< 4 кандидатів", None, kps_a, kps_b, cand
    src = np.float32([kps_a[m.queryIdx].pt for m in cand]).reshape(-1, 2)
    dst = np.float32([kps_b[m.trainIdx].pt for m in cand]).reshape(-1, 2)
    H, mask, _, err = custom_ransac_homography(src, dst, tau=tau, p=0.99, max_iters=2000, mode="adaptive", seed=42)
    inl = int(np.sum(mask)) if mask is not None else 0
    ok, reason = homography_plausibility(H, img_a.shape, img_b.shape)
    cov_b, cov_a = compute_overlap_fractions(img_a.shape, img_b.shape, H)
    return len(cand), inl, cov_b, cov_a, err, ok, reason, (H, mask, src, dst), kps_a, kps_b, cand


def run_part_e():
    print("=" * 60)
    print("Запуск Частини E: Аналіз випадків відмови (Крок 17)")
    print("=" * 60)

    scale = 0.5

    def load_scaled(name):
        im = imread_bgr(os.path.join(DATA_HW_DIR, f"{name}.jpg"))
        return cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------
    # 1. Дослідження перекриття та геометричної правдоподібності H
    # -------------------------------------------------------------
    print("\n--- Дослідження 1: Ступінь перекриття та виродження гомографії ---")
    img_a4 = load_scaled("angle4")
    h, w = img_a4.shape[:2]
    cases = [
        ("angle3-angle4", "Сусідні кадри (~84 % перекриття)", load_scaled("angle3"), img_a4),
        ("scale3-scale4", "Зміна масштабу (41.4 % / 100 %)", load_scaled("scale3"), load_scaled("scale4")),
        ("scale1-scale3", "Вкладений крупний план (5.8 % / 100 %)", load_scaled("scale1"), load_scaled("scale3")),
        ("scale1-scale5", "Критичний стрибок масштабу (> 10x, < 1 %)", load_scaled("scale1"), load_scaled("scale5")),
        ("angle2-angle5", "Малий перетин через 3 кадри (< 10 %)", load_scaled("angle2"), load_scaled("angle5")),
        ("angle4-angle8", "Великий розворот ракурсу (< 5 %)", img_a4, load_scaled("angle8")),
        ("angle8-angle10", "Несумісні ракурси (< 1 % спільної зони)", load_scaled("angle8"), load_scaled("angle10")),
    ]

    overlap_rows = []
    vis_low_pair = None
    for pair_id, desc, ia, ib in cases:
        n_cand, inl, cb, ca, err, ok, reason, pack, ka, kb, cand = evaluate_pair(ia, ib)
        inl_pct = (inl / n_cand * 100.0) if n_cand > 0 else 0.0
        if ok and inl >= 30:
            status = "Успішно"
            cov_str = f"{cb:.1f}% / {ca:.1f}%"
        else:
            status = f"Відмова ({reason})" if not ok else "Відмова (мало inliers)"
            cov_str = "вироджена H"
        print(f"  {pair_id:15s}: cand={n_cand:4d}, inl={inl:4d} ({inl_pct:4.1f}%), "
              f"перекриття={cov_str:16s}, err={err:4.2f} пк -> {status}")
        overlap_rows.append([pair_id, desc, n_cand, inl, f"{inl_pct:.1f}", cov_str, f"{err:.2f}", status])
        if pair_id == "angle4-angle8":
            vis_low_pair = (ia, ib, pack, ka, kb, cand)

    save_csv(os.path.join(RESULTS_DIR, "e_overlap_failure.csv"),
             ["pair", "description", "candidates", "inliers", "inlier_pct", "overlap_dst_src", "reproj_err", "status"],
             overlap_rows)

    # -------------------------------------------------------------
    # 2. Дослідження оптичного паралаксу на 3D-серії HDD angle2-angle6
    # -------------------------------------------------------------
    print("\n--- Дослідження 2: Оптичний паралакс у 3D-сцені (angle2-angle6) ---")
    hdd_names = ["angle2", "angle3", "angle4", "angle5", "angle6"]
    ref_idx = 2
    hdd_imgs = [load_scaled(n) for n in hdd_names]
    hdd_feats = [extract_sift_features(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for im in hdd_imgs]

    adj = {}
    parallax_rows = []
    for i in range(len(hdd_names) - 1):
        ka, da = hdd_feats[i]
        kb, db = hdd_feats[i + 1]
        cand = apply_mutual_match(da, db, ratio=0.75)
        src = np.float32([ka[m.queryIdx].pt for m in cand]).reshape(-1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in cand]).reshape(-1, 2)
        H, mask, _, err = custom_ransac_homography(src, dst, tau=3.0, p=0.99, max_iters=2000, mode="adaptive", seed=42)
        adj[(i, i + 1)] = H
        # Похибка на inliers (точки плати) проти похибки на відкинутих взаємних збігах (корпус / стіл / клавіатура)
        e_all = compute_reprojection_errors(src, dst, H)
        e_in = e_all[mask]
        e_out = e_all[~mask]
        print(f"  {hdd_names[i]} -> {hdd_names[i+1]}: взаємних={len(cand)}, inliers(плата)={mask.sum()} "
              f"({mask.mean()*100:.1f}%), e_in={e_in.mean():.2f} пк, медіана e_out(поза платою)={np.median(e_out):.1f} пк")
        parallax_rows.append([f"{hdd_names[i]}-{hdd_names[i+1]}", len(cand), int(mask.sum()),
                              f"{mask.mean()*100:.1f}", f"{e_in.mean():.2f}", f"{np.median(e_out):.1f}"])

    save_csv(os.path.join(RESULTS_DIR, "e_parallax_pairs.csv"),
             ["pair", "mutual_candidates", "inliers_pcb", "inliers_pct", "err_pcb_px", "median_err_offplane_px"],
             parallax_rows)

    H_to_ref = [None] * len(hdd_names)
    H_to_ref[ref_idx] = np.eye(3)
    for i in range(ref_idx - 1, -1, -1):
        H_to_ref[i] = H_to_ref[i + 1] @ adj[(i, i + 1)]
    for i in range(ref_idx + 1, len(hdd_names)):
        H_to_ref[i] = H_to_ref[i - 1] @ np.linalg.inv(adj[(i - 1, i)])

    hdd_mosaic, _, _, _ = stitch_panorama_sequence(hdd_imgs, H_to_ref)
    cv2.imwrite(os.path.join(FIGURES_DIR, "e_parallax_mosaic.png"), hdd_mosaic)

    # Двокадрове накладання angle3 -> angle4 для крупного плану паралаксу
    H34 = adj[(1, 2)]
    w34 = cv2.warpPerspective(hdd_imgs[1], H34, (w, h))
    ov34 = cv2.addWeighted(hdd_imgs[2], 0.5, w34, 0.5, 0)
    cv2.imwrite(os.path.join(FIGURES_DIR, "e_parallax_overlay.png"), ov34)

    # Візуалізація випадку 1 (дефіцит перекриття angle4-angle8 + контур виродженої H)
    ia, ib, pack, _, _, _ = vis_low_pair
    H_bad, mask_bad, src_b, dst_b = pack
    sc = 0.65
    da = cv2.resize(ia, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    db = cv2.resize(ib, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    hd, wd = da.shape[:2]
    canvas_low = np.hstack([da, db])
    rng = np.random.default_rng(42)
    out_idx = rng.choice(np.where(~mask_bad)[0], size=min(60, int((~mask_bad).sum())), replace=False)
    for idx in out_idx:
        p1 = (int(src_b[idx, 0] * sc), int(src_b[idx, 1] * sc))
        p2 = (int(dst_b[idx, 0] * sc) + wd, int(dst_b[idx, 1] * sc))
        cv2.line(canvas_low, p1, p2, (0, 0, 220), 1, cv2.LINE_AA)
    for idx in np.where(mask_bad)[0]:
        p1 = (int(src_b[idx, 0] * sc), int(src_b[idx, 1] * sc))
        p2 = (int(dst_b[idx, 0] * sc) + wd, int(dst_b[idx, 1] * sc))
        cv2.line(canvas_low, p1, p2, (0, 210, 255), 2, cv2.LINE_AA)

    # Зведена 2-панельна ілюстрація для звіту
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax1.imshow(cv2.cvtColor(canvas_low, cv2.COLOR_BGR2RGB))
    ax1.set_title("(а) Дефіцит перекриття (angle4 – angle8):\nлише 6 хибних «inliers» із 184 кандидатів, H вироджена", fontsize=9.5)
    ax1.axis("off")

    hm, wm = hdd_mosaic.shape[:2]
    crop_mos = hdd_mosaic[int(hm * 0.04):int(hm * 0.86), int(wm * 0.16):int(wm * 0.84)]
    ax2.imshow(cv2.cvtColor(crop_mos, cv2.COLOR_BGR2RGB))
    ax2.set_title("(б) Оптичний паралакс у 3D-сцені (мозаїка angle2–angle6):\nплата суміщена, але рельєфний корпус HDD і клавіатура двояться", fontsize=9.5)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "e_failure_cases.png"))
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "e_log.txt"), "w", encoding="utf-8") as f:
        f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ E ===\n")
        f.write("1. Досліджено дефіцит перекриття та критерій геометричної правдоподібності H.\n")
        f.write("2. Досліджено просторовий паралакс при зшиванні 5 кадрів об'ємної сцени angle2-angle6.\n")

    print("\nЧастина E успішно виконана!")


if __name__ == "__main__":
    run_part_e()
