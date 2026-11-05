"""
Лабораторна робота № 4. Частина D: Побудова панорами (Кроки 13-16).

Вибір сцени. Гомографія точно описує зв'язок двох кадрів лише у двох випадках:
(1) камера обертається навколо оптичного центру (без зміщення), або (2) сцена плоска.
Серія angle1-angle10 знята обходом камери навколо об'ємного HDD, тому вона не задовольняє
жодну з умов (див. Частину E). Для коректної панорами використано серію кадрів
lab02/data/corners: шахова дошка, що лежить на столі, знята з різних положень камери.
Стіл і дошка утворюють одну площину, тому гомографія між кадрами фізично точна.

- Крок 13: Гомографії сусідніх пар (власний RANSAC), опорний -- центральний кадр, ланцюгове множення;
- Крок 14: Розмір полотна за проєкціями кутів кадрів, додатний зсув T;
- Крок 15: cv2.warpPerspective + лінійний перехід (feathering) у зонах перекриття;
- Крок 16: Середня похибка репроєкції на узгоджених точках усіх пар + перевірка накопичення
           похибки ланцюга (порівняння ланцюгової та прямої гомографії до опорного кадру).
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    ROOT_DIR, RESULTS_DIR, FIGURES_DIR,
    imread_bgr, save_csv,
    extract_sift_features, apply_mutual_match,
    compute_reprojection_errors, custom_ransac_homography, stitch_panorama_sequence,
    homography_corner_distance
)

DATA_PANO_DIR = os.path.join(ROOT_DIR, "lab02", "data", "corners")
FRAME_NAMES = ["corner_7", "corner_6", "corner_4", "corner_5", "corner_3"]  # зліва направо
REF_IDX = 2  # corner_4
SCALE = 0.5


def load_frames(names, data_dir, scale):
    imgs, feats = [], []
    for name in names:
        img = imread_bgr(os.path.join(data_dir, f"{name}.jpg"))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        kps, desc = extract_sift_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        imgs.append(img)
        feats.append((kps, desc))
        print(f"Кадр {name}: {img.shape[1]}x{img.shape[0]}, SIFT-точок: {len(kps)}")
    return imgs, feats


def estimate_pair(feat_a, feat_b, ratio=0.75, tau=3.0, seed=42):
    """Кандидати Лоу + взаємна перевірка + власний RANSAC. Повертає H (a -> b), маску, кандидатів, точки та похибку."""
    kps_a, desc_a = feat_a
    kps_b, desc_b = feat_b
    cand = apply_mutual_match(desc_a, desc_b, ratio=ratio)
    src = np.float32([kps_a[m.queryIdx].pt for m in cand]).reshape(-1, 2)
    dst = np.float32([kps_b[m.trainIdx].pt for m in cand]).reshape(-1, 2)
    H, mask, _, err = custom_ransac_homography(src, dst, tau=tau, p=0.99, max_iters=3000, mode="adaptive", seed=seed)
    return H, mask, cand, src, dst, err


def chain_to_reference(adjacent, n_frames, ref_idx):
    """
    adjacent[(i, i+1)] -- гомографія кадр i -> кадр i+1.
    Ліворуч від опорного: H_{i->ref} = H_{i+1->ref} @ H_{i->i+1};
    праворуч:             H_{i->ref} = H_{i-1->ref} @ inv(H_{i-1->i}).
    """
    H_to_ref = [None] * n_frames
    H_to_ref[ref_idx] = np.eye(3)
    for i in range(ref_idx - 1, -1, -1):
        H_to_ref[i] = H_to_ref[i + 1] @ adjacent[(i, i + 1)]
    for i in range(ref_idx + 1, n_frames):
        H_to_ref[i] = H_to_ref[i - 1] @ np.linalg.inv(adjacent[(i - 1, i)])
    return [H / H[2, 2] for H in H_to_ref]


def run_part_d():
    print("=" * 60)
    print("Запуск Частини D: Побудова панорами з 5 кадрів (Кроки 13-16)")
    print("=" * 60)

    n = len(FRAME_NAMES)
    images, feats = load_frames(FRAME_NAMES, DATA_PANO_DIR, SCALE)
    h, w = images[0].shape[:2]

    # Крок 13 + 16: сусідні пари
    adjacent, pair_rows, all_err = {}, [], []
    for i in range(n - 1):
        H, mask, cand, src, dst, err = estimate_pair(feats[i], feats[i + 1])
        adjacent[(i, i + 1)] = H
        e = compute_reprojection_errors(src[mask], dst[mask], H)
        all_err.append(e)
        inl = int(mask.sum())
        print(f"{FRAME_NAMES[i]} -> {FRAME_NAMES[i+1]}: кандидатів={len(cand)}, inliers={inl} "
              f"({inl/len(cand)*100:.1f}%), e: mean={e.mean():.2f}, std={e.std():.2f}, max={e.max():.2f}")
        pair_rows.append([f"{FRAME_NAMES[i]}-{FRAME_NAMES[i+1]}", len(cand), inl, f"{inl/len(cand)*100:.1f}",
                          f"{e.mean():.2f}", f"{e.std():.2f}", f"{e.max():.2f}"])
    save_csv(os.path.join(RESULTS_DIR, "d_pair_quality.csv"),
             ["pair", "candidates", "inliers", "inlier_pct", "mean_reproj_err", "std_reproj_err", "max_reproj_err"],
             pair_rows)
    all_err = np.concatenate(all_err)
    mean_all = float(all_err.mean())
    print(f"[Крок 16] Середня похибка на всіх {len(all_err)} inliers: {mean_all:.2f} пк "
          f"(медіана {np.median(all_err):.2f}, 95-й перцентиль {np.percentile(all_err, 95):.2f})")

    H_to_ref = chain_to_reference(adjacent, n, REF_IDX)

    # Перевірка накопичення похибки ланцюга для крайніх кадрів (глибина ланцюга 2):
    # пряма гомографія крайній -> опорний vs добуток двох попарних
    chain_rows = []
    for i in (0, n - 1):
        H_dir, m_dir, cand_d, src_d, dst_d, _ = estimate_pair(feats[i], feats[REF_IDX])
        corner_d = homography_corner_distance(H_to_ref[i], H_dir / H_dir[2, 2], (h, w))
        e_chain = compute_reprojection_errors(src_d[m_dir], dst_d[m_dir], H_to_ref[i])
        e_direct = compute_reprojection_errors(src_d[m_dir], dst_d[m_dir], H_dir)
        print(f"Ланцюг {FRAME_NAMES[i]} -> {FRAME_NAMES[REF_IDX]}: зміщення кутів (ланцюг vs пряма) = {corner_d:.2f} пк; "
              f"похибка на {int(m_dir.sum())} прямих inliers: ланцюг {e_chain.mean():.2f} пк, пряма {e_direct.mean():.2f} пк")
        chain_rows.append([f"{FRAME_NAMES[i]}-{FRAME_NAMES[REF_IDX]}", int(m_dir.sum()),
                           f"{e_direct.mean():.2f}", f"{e_chain.mean():.2f}", f"{corner_d:.2f}"])
    save_csv(os.path.join(RESULTS_DIR, "d_chain_check.csv"),
             ["pair", "direct_inliers", "err_direct", "err_chain", "corner_diff_px"], chain_rows)

    # Кроки 14-15
    pano_crop, pano_full, canvas, offset = stitch_panorama_sequence(images, H_to_ref)
    print(f"Полотно: {canvas[0]}x{canvas[1]}, зсув T=({offset[0]}, {offset[1]}); після обрізання: "
          f"{pano_crop.shape[1]}x{pano_crop.shape[0]}")
    cv2.imwrite(os.path.join(FIGURES_DIR, "d_panorama.png"), pano_crop)
    cv2.imwrite(os.path.join(FIGURES_DIR, "d_panorama_full_canvas.png"), pano_full)

    # Порівняння: без лінійного переходу (останній кадр перекриває попередні) -- для демонстрації швів
    T = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]], dtype=np.float64)
    naive = np.zeros((canvas[1], canvas[0], 3), np.uint8)
    for img, H in zip(images, H_to_ref):
        wimg = cv2.warpPerspective(img, T @ H, canvas)
        wm = cv2.warpPerspective(np.full((h, w), 255, np.uint8), T @ H, canvas) > 0
        naive[wm] = wimg[wm]
    cv2.imwrite(os.path.join(FIGURES_DIR, "d_panorama_naive.png"), naive)

    # Межі кадрів на полотні (для рисунка)
    outline = pano_full.copy()
    colors = [(230, 80, 0), (0, 160, 255), (0, 0, 230), (200, 0, 200), (0, 170, 0)]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    for k, H in enumerate(H_to_ref):
        poly = cv2.perspectiveTransform(corners, T @ H).astype(np.int32)
        cv2.polylines(outline, [poly], True, colors[k], 6, cv2.LINE_AA)
    cv2.imwrite(os.path.join(FIGURES_DIR, "d_panorama_outlines.png"), outline)

    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 2.2])
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax.set_title(FRAME_NAMES[i] + (" (опорний)" if i == REF_IDX else ""), fontsize=9,
                     color=np.array(colors[i][::-1]) / 255.0)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(cv2.cvtColor(outline, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Панорама {canvas[0]}×{canvas[1]} пк з межами кадрів; середня похибка на inliers {mean_all:.2f} пк",
                 fontsize=10.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "d_panorama_overview.png"))
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "d_log.txt"), "w", encoding="utf-8") as f:
        f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ D ===\n")
        f.write(f"Серія: {FRAME_NAMES} (lab02/data/corners, масштаб {SCALE}: {w}x{h}), опорний: {FRAME_NAMES[REF_IDX]}.\n")
        f.write(f"Полотно: {canvas[0]}x{canvas[1]}, зсув: ({offset[0]}, {offset[1]}).\n")
        f.write(f"Середня похибка на {len(all_err)} inliers усіх пар: {mean_all:.2f} пк.\n")
    save_csv(os.path.join(RESULTS_DIR, "d_summary.csv"),
             ["frames", "frame_w", "frame_h", "canvas_w", "canvas_h", "offset_x", "offset_y", "total_inliers", "mean_err",
              "median_err", "p95_err"],
             [[n, w, h, canvas[0], canvas[1], offset[0], offset[1], len(all_err), f"{mean_all:.2f}",
               f"{np.median(all_err):.2f}", f"{np.percentile(all_err, 95):.2f}"]])
    print("\nЧастина D успішно виконана!")


if __name__ == "__main__":
    run_part_d()
