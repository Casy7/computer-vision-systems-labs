"""
Лабораторна робота № 4. Частина A: Зіставлення ознак (Кроки 1-5).
- Крок 1: Зчитування пари сусідніх кадрів, обчислення SIFT-дескрипторів;
- Крок 2: BFMatcher (NORM_L2), knnMatch з k=2;
- Крок 3: Критерій відношення відстаней Лоу для T = 0.5, 0.6, 0.7, 0.8, 0.9, 0.95;
- Крок 4: Взаємна перевірка зіставлень (Cross-check);
- Крок 5: Візуалізація сирих зіставлень із наочно помітними хибними лініями.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    DATA_HW_DIR, RESULTS_DIR, FIGURES_DIR,
    imread_bgr, save_csv,
    extract_sift_features, match_knn, apply_ratio_test, apply_mutual_match,
    compute_overlap_fractions
)


def run_part_a():
    print("=" * 60)
    print("Запуск Частини A: Зіставлення локальних ознак (Кроки 1-5)")
    print("=" * 60)

    # Кадри для дослідження:
    # 1. Основна пара панорамної серії: angle4 та angle5
    # 2. Додаткова пара з перекриттям ~40.5%: scale3 та scale4
    pairs_to_test = [
        ("angle4", "angle5", os.path.join(DATA_HW_DIR, "angle4.jpg"), os.path.join(DATA_HW_DIR, "angle5.jpg")),
        ("scale3", "scale4", os.path.join(DATA_HW_DIR, "scale3.jpg"), os.path.join(DATA_HW_DIR, "scale4.jpg"))
    ]

    ratio_thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
    all_ratio_rows = []
    pair_rows = []

    for name_a, name_b, path_a, path_b in pairs_to_test:
        print(f"\n--- Дослідження пари: {name_a} <-> {name_b} ---")
        img_a_bgr = imread_bgr(path_a)
        img_b_bgr = imread_bgr(path_b)
        gray_a = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2GRAY)

        # Крок 1: SIFT-дескриптори
        kps_a, desc_a = extract_sift_features(gray_a)
        kps_b, desc_b = extract_sift_features(gray_b)
        print(f"Кількість SIFT-точок: {name_a}={len(kps_a)}, {name_b}={len(kps_b)}")

        # Крок 2: k-NN зіставлення (k=2) в обох напрямках
        print("Виклик BFMatcher.knnMatch (в обох напрямках один раз)...")
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches_ab = bf.knnMatch(desc_a, desc_b, k=2)
        matches_ba = bf.knnMatch(desc_b, desc_a, k=2)
        total_raw = len(matches_ab)
        print(f"Загальна кількість первинних сирих кандидатів: {total_raw}")

        # Крок 3 та 4: Критерій відношення Лоу та взаємна перевірка для сітки T
        counts_for_T = {}
        mutual_for_T = {}
        for T in ratio_thresholds:
            good = apply_ratio_test(matches_ab, ratio=T)
            mut = apply_mutual_match(desc_a, desc_b, ratio=T, matches_12=matches_ab, matches_21=matches_ba)
            counts_for_T[T] = len(good)
            mutual_for_T[T] = len(mut)
            print(f"  Поріг T={T:.2f}: Лоу={len(good):5d} ({len(good)/total_raw*100:5.2f}%), "
                  f"Взаємні={len(mut):5d} (відкинуто {len(good)-len(mut):5d})")

            all_ratio_rows.append([
                f"{name_a}-{name_b}",
                len(kps_a),
                len(kps_b),
                total_raw,
                f"{T:.2f}",
                counts_for_T[T],
                f"{counts_for_T[T]/total_raw*100:.2f}",
                mutual_for_T[T],
                f"{mutual_for_T[T]/counts_for_T[T]*100:.1f}" if counts_for_T[T] > 0 else "0.0"
            ])

        # Крок 5: Візуалізація для пари angle4-angle5
        if name_a == "angle4":
            print("\nГенерація ілюстрацій для Кроку 5...")

            # 1. Сирі кандидати: найближчий сусід без жодної фільтрації (до тесту Лоу)
            raw_nn = [pair[0] for pair in matches_ab if len(pair) >= 1]
            rng = np.random.default_rng(42)
            sample_raw = [raw_nn[i] for i in rng.choice(len(raw_nn), size=min(150, len(raw_nn)), replace=False)]

            scale_disp = 0.35
            disp_a = cv2.resize(img_a_bgr, (0, 0), fx=scale_disp, fy=scale_disp, interpolation=cv2.INTER_AREA)
            disp_b = cv2.resize(img_b_bgr, (0, 0), fx=scale_disp, fy=scale_disp, interpolation=cv2.INTER_AREA)
            
            disp_kps_a = [cv2.KeyPoint(kp.pt[0]*scale_disp, kp.pt[1]*scale_disp, kp.size*scale_disp) for kp in kps_a]
            disp_kps_b = [cv2.KeyPoint(kp.pt[0]*scale_disp, kp.pt[1]*scale_disp, kp.size*scale_disp) for kp in kps_b]

            vis_raw = cv2.drawMatches(
                disp_a, disp_kps_a,
                disp_b, disp_kps_b,
                sample_raw, None,
                matchColor=(0, 0, 240),  # яскраво-червоні лінії (BGR)
                singlePointColor=(255, 120, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(os.path.join(FIGURES_DIR, "a_raw_matches.png"), vis_raw)

            # 2. Фільтровані взаємні зіставлення при робочому T=0.75
            mut_75 = apply_mutual_match(desc_a, desc_b, ratio=0.75, matches_12=matches_ab, matches_21=matches_ba)
            sample_mut = [mut_75[i] for i in rng.choice(len(mut_75), size=min(120, len(mut_75)), replace=False)]

            vis_filtered = cv2.drawMatches(
                disp_a, disp_kps_a,
                disp_b, disp_kps_b,
                sample_mut, None,
                matchColor=(0, 200, 0),  # яскраво-зелені лінії (BGR)
                singlePointColor=(0, 200, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(os.path.join(FIGURES_DIR, "a_mutual_matches.png"), vis_filtered)

            # 3. Графік залежності кандидатів від T (логарифмічна вісь Y: значення
            #    змінюються на три порядки, у лінійній шкалі точки T <= 0.7 зливаються з нулем)
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            t_vals = ratio_thresholds
            c_angle = [counts_for_T[t] for t in t_vals]
            m_angle = [mutual_for_T[t] for t in t_vals]

            ax.semilogy(t_vals, c_angle, "o-", color="#d95f02", label="Критерій відношення Лоу ($d_1 < T \\cdot d_2$)", lw=2.0, ms=6)
            ax.semilogy(t_vals, m_angle, "s--", color="#1b9e77", label="Критерій Лоу + взаємна перевірка", lw=2.0, ms=6)
            for t, c, mm in zip(t_vals, c_angle, m_angle):
                ax.annotate(f"{c}", (t, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#d95f02", fontweight="bold")
                ax.annotate(f"{mm}", (t, mm), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8, color="#1b9e77", fontweight="bold")
            ax.set_xlim(0.47, 0.98)
            ax.set_ylim(2.2, 2.5e4)
            ax.set_xlabel("Поріг відношення відстаней $T$")
            ax.set_ylabel("Кількість пар-кандидатів, од. (лог. шкала)")
            ax.set_title("Кількість кандидатів від порога $T$ (angle4 – angle5)")
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
            ax.legend(loc="upper left")
            plt.savefig(os.path.join(FIGURES_DIR, "a_ratio_curve.png"))
            plt.close(fig)

        # Оцінка геометричного перекриття пари (через RANSAC-гомографію домінантної площини)
        cand_80 = apply_ratio_test(matches_ab, ratio=0.80)
        s80 = np.float32([kps_a[m.queryIdx].pt for m in cand_80]).reshape(-1, 1, 2)
        d80 = np.float32([kps_b[m.trainIdx].pt for m in cand_80]).reshape(-1, 1, 2)
        H_ref, _ = cv2.findHomography(s80, d80, cv2.RANSAC, 3.0)
        cov_b, cov_a = compute_overlap_fractions(gray_a.shape, gray_b.shape, H_ref)
        print(f"Перекриття (за H домінантної площини): покриття {name_b} = {cov_b:.1f}%, видима частка {name_a} = {cov_a:.1f}%")
        pair_rows.append([f"{name_a}-{name_b}", len(kps_a), len(kps_b), f"{cov_b:.1f}", f"{cov_a:.1f}"])

    save_csv(os.path.join(RESULTS_DIR, "a_pairs.csv"),
             ["pair", "kp_src", "kp_dst", "overlap_cov_dst_pct", "overlap_cov_src_pct"], pair_rows)

    # Збереження CSV-таблиці
    csv_header = ["pair", "kp_src", "kp_dst", "raw_matches", "ratio_T", "candidates", "cand_pct", "mutual_candidates", "mutual_retention_pct"]
    save_csv(os.path.join(RESULTS_DIR, "a_ratio_test.csv"), csv_header, all_ratio_rows)

    # Текстовий журнал
    with open(os.path.join(RESULTS_DIR, "a_log.txt"), "w", encoding="utf-8") as f:
        f.write("=== ЖУРНАЛ ВИКОНАННЯ ЧАСТИНИ A ===\n")
        f.write(f"Досліджено пари angle4-angle5 та scale3-scale4.\n")
        f.write(f"Поріг T від 0.50 до 0.95 демонструє стрімке нелінійне зростання кандидатів.\n")
        f.write(f"Взаємна перевірка відсікає асиметричні хибні відповідники, стабілізуючи вибірку.\n")

    print("\nЧастина A успішно виконана!")


if __name__ == "__main__":
    run_part_a()
