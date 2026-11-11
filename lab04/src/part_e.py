"""
Лабораторна робота № 4. Частина E: Аналіз випадків відмови (Крок 17).
Відтворення та кількісний аналіз двох класичних режимів відмови гомографічного зшивання:

1. Дефіцит геометричного перекриття (< 15 %) та великий розворот ракурсу:
   - порівняння сусідніх пар панорами (pano_1-pano_2, pano_2-pano_3, pano_4-pano_5 із видимою часткою 37-49 %),
     пар через один кадр (pano_1-pano_3, pano_3-pano_5) та рознесених пар із дефіцитом перекриття < 15 %
     (pano_2-pano_5, pano_1-pano_5), де кількість узгоджених точок падає до 8-18 випадкових збігів,
     а оцінена матриця H стає геометрично виродженою (перетин лінії горизонту).

2. Просторовий паралакс при бічному зміщенні оптичного центра камери в 3D-сцені:
   - спеціально знята пара кадрів err_pano_1 та err_pano_2 із поперечним кроком камери вздовж стіни з полицями:
     RANSAC фіксує домінантну площину дальньої стіни та настінного годинника (похибка ~1.09-1.41 пк),
     тоді як виступаючі об'ємні полиці, ваза з сухоцвітами, чашка, світильник-бра, чайники та край холодильника
     мають меншу глибину Z і зазнають сильного паралактичного розщеплення (двоїння контурів).
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
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

    def load_img(name):
        return imread_bgr(os.path.join(DATA_DIR, f"{name}.jpg"))

    imgs = {f"pano_{i}": load_img(f"pano_{i}") for i in range(1, 6)}
    imgs["err_pano_1"] = load_img("err_pano_1")
    imgs["err_pano_2"] = load_img("err_pano_2")
    h, w = imgs["pano_1"].shape[:2]

    # -------------------------------------------------------------
    # 1. Дослідження перекриття та геометричної правдоподібності H
    # -------------------------------------------------------------
    print("\n--- Дослідження 1: Ступінь перекриття та виродження гомографії ---")
    cases = [
        ("pano_1-pano_2", "Сусідні кадри (41.3 % видимої площі)", imgs["pano_1"], imgs["pano_2"]),
        ("pano_2-pano_3", "Сусідні кадри (48.7 % видимої площі)", imgs["pano_2"], imgs["pano_3"]),
        ("pano_4-pano_5", "Сусідні кадри (36.9 % видимої площі)", imgs["pano_4"], imgs["pano_5"]),
        ("pano_1-pano_3", "Через 1 кадр (1.8 % проєкції кадру)", imgs["pano_1"], imgs["pano_3"]),
        ("pano_3-pano_5", "Через 1 кадр (розворот > 50°)", imgs["pano_3"], imgs["pano_5"]),
        ("pano_2-pano_5", "Дефіцит перекриття (< 10 % зони)", imgs["pano_2"], imgs["pano_5"]),
        ("pano_1-pano_5", "Протилежні краї панорами (0 %)", imgs["pano_1"], imgs["pano_5"]),
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
        if pair_id == "pano_2-pano_5":
            vis_low_pair = (ia, ib, pack, ka, kb, cand)

    save_csv(os.path.join(RESULTS_DIR, "e_overlap_failure.csv"),
             ["pair", "description", "candidates", "inliers", "inlier_pct", "overlap_dst_src", "reproj_err", "status"],
             overlap_rows)

    # -------------------------------------------------------------
    # 2. Дослідження оптичного паралаксу при бічному зміщенні камери (err_pano_1, err_pano_2)
    # -------------------------------------------------------------
    print("\n--- Дослідження 2: Оптичний паралакс при зміщенні камери (err_pano_1 - err_pano_2) ---")
    feats = {k: extract_sift_features(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
             for k, im in imgs.items() if k in ("err_pano_1", "err_pano_2", "pano_1", "pano_2")}

    parallax_configs = [
        ("err_pano_1-err_pano_2 (взаємні)", "err_pano_1", "err_pano_2", 0.75, True),
        ("err_pano_1-err_pano_2 (Лоу 0.80)", "err_pano_1", "err_pano_2", 0.80, False),
        ("err_pano_1-pano_1 (взаємні)", "err_pano_1", "pano_1", 0.75, True),
        ("err_pano_2-pano_1 (взаємні)", "err_pano_2", "pano_1", 0.75, True),
    ]

    parallax_rows = []
    H_err_80 = None
    for label, na, nb, ratio, use_mut in parallax_configs:
        ka, da = feats[na]
        kb, db = feats[nb]
        if use_mut:
            cand = apply_mutual_match(da, db, ratio=ratio)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            cand = apply_ratio_test(bf.knnMatch(da, db, k=2), ratio=ratio)
        src = np.float32([ka[m.queryIdx].pt for m in cand]).reshape(-1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in cand]).reshape(-1, 2)
        H, mask, _, err = custom_ransac_homography(src, dst, tau=3.0, p=0.99, max_iters=2000, mode="adaptive", seed=42)
        if not use_mut:
            H_err_80 = H
        e_all = compute_reprojection_errors(src, dst, H)
        e_in = e_all[mask]
        e_out = e_all[~mask]
        print(f"  {label:32s}: cand={len(cand):3d}, inliers(стіна)={mask.sum():3d} "
              f"({mask.mean()*100:.1f}%), e_in={e_in.mean():.2f} пк, медіана e_out(полиці)={np.median(e_out):.1f} пк")
        parallax_rows.append([label, len(cand), int(mask.sum()),
                              f"{mask.mean()*100:.1f}", f"{e_in.mean():.2f}", f"{np.median(e_out):.1f}"])

    save_csv(os.path.join(RESULTS_DIR, "e_parallax_pairs.csv"),
             ["pair", "mutual_candidates", "inliers_pcb", "inliers_pct", "err_pcb_px", "median_err_offplane_px"],
             parallax_rows)

    # Побудова двокадрової мозаїки та напівпрозорого накладання err_pano_1 -> err_pano_2
    e1, e2 = imgs["err_pano_1"], imgs["err_pano_2"]
    err_mosaic, _, _, _ = stitch_panorama_sequence([e1, e2], [H_err_80, np.eye(3)])
    cv2.imwrite(os.path.join(FIGURES_DIR, "e_parallax_mosaic.png"), err_mosaic)

    w12 = cv2.warpPerspective(e1, H_err_80, (w, h))
    m12 = (w12.sum(axis=2) > 0)
    ov12 = e2.copy()
    ov12[m12] = cv2.addWeighted(w12[m12], 0.5, e2[m12], 0.5, 0)
    cv2.imwrite(os.path.join(FIGURES_DIR, "e_parallax_overlay.png"), ov12)

    # Візуалізація випадку 1 (дефіцит перекриття pano_2-pano_5)
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax1.imshow(cv2.cvtColor(canvas_low, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"(а) Дефіцит перекриття (pano_2 – pano_5, < 10 %):\nлише {int(mask_bad.sum())} хибних «inliers» зі {len(src_b)} кандидата, H вироджена", fontsize=9.5)
    ax1.axis("off")

    crop_ov = ov12[int(h * 0.02):int(h * 0.82), int(w * 0.05):int(w * 0.96)]
    ax2.imshow(cv2.cvtColor(crop_ov, cv2.COLOR_BGR2RGB))
    ax2.set_title("(б) Паралакс при зміщенні камери (err_pano_1 – err_pano_2):\nгодинник на стіні суміщено, а виступаючі полиці й ваза двояться", fontsize=9.5)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "e_failure_cases.png"))
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "e_log.txt"), "w", encoding="utf-8") as f:
        f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ E ===\n")
        f.write("1. Досліджено дефіцит перекриття на парах панорамної серії pano_1..pano_5.\n")
        f.write("2. Досліджено просторовий паралакс при бічному зміщенні камери на парі err_pano_1-err_pano_2.\n")

    print("\nЧастина E успішно виконана!")


if __name__ == "__main__":
    run_part_e()
