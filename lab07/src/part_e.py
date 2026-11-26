"""Частина E Лабораторної роботи №7: Детектор пішоходів HOG+SVM, власний NMS, зіставлення з еталоном та крива Precision--Recall / AP (Кроки 11--16)."""

import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    PART_E_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
)

np.random.seed(42)


def parse_voc_xml(xml_path: Path):
    """Зчитує еталонні рамки класу person у форматі [x1, y1, x2, y2] з Pascal VOC XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name != "person":
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = float(bb.findtext("xmin"))
        y1 = float(bb.findtext("ymin"))
        x2 = float(bb.findtext("xmax"))
        y2 = float(bb.findtext("ymax"))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return filename, np.array(boxes, dtype=np.float64).reshape(-1, 4)


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Обчислює IoU для двох прямокутних рамок [x1, y1, x2, y2]."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5):
    """
    Крок 13: Власна реалізація немаксимального придушення (NMS).
    Сортує рамки за спаданням впевненості та видаляє всі рамки з IoU > iou_thresh від уже прийнятої.
    """
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        filtered = []
        for j in rest:
            if bbox_iou(boxes[i], boxes[int(j)]) <= iou_thresh:
                filtered.append(int(j))
        order = np.array(filtered, dtype=int)

    keep_idx = np.array(keep, dtype=int)
    return boxes[keep_idx], scores[keep_idx]


def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray):
    """Крок 16: Монотонна незростаюча огинаюча точності та обчислення площі під кривою (AP)."""
    mrec = np.concatenate(([0.0], recalls, [recalls[-1] if len(recalls) > 0 else 0.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx_change = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx_change + 1] - mrec[idx_change]) * mpre[idx_change + 1]))
    return ap, mrec, mpre


def evaluate_detections_pr_ap(all_dets, gt_by_image, iou_thresh: float = 0.5, sort_by_conf: bool = True):
    """
    Кроки 14--16: Зіставлення з еталоном (кожен еталон не більше одного разу),
    побудова кривої Precision--Recall та обчислення AP через монотонну огинаючу точності.
    """
    dets = list(all_dets)
    if sort_by_conf:
        dets.sort(key=lambda d: d["score"], reverse=True)
    else:
        rng = np.random.default_rng(42)
        rng.shuffle(dets)

    total_gt = sum(len(b) for b in gt_by_image.values())
    matched_flags = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_by_image.items()}

    tp_flags = np.zeros(len(dets), dtype=np.float64)
    fp_flags = np.zeros(len(dets), dtype=np.float64)
    det_records = []

    for idx, d in enumerate(dets):
        img_id = d["img_id"]
        box = d["box"]
        gt_boxes = gt_by_image[img_id]
        best_iou = 0.0
        best_gt = -1
        for g_idx, g_box in enumerate(gt_boxes):
            ov = bbox_iou(box, g_box)
            if ov > best_iou:
                best_iou = ov
                best_gt = g_idx

        is_tp = (best_iou >= iou_thresh) and (best_gt >= 0) and (not matched_flags[img_id][best_gt])
        if is_tp:
            matched_flags[img_id][best_gt] = True
            tp_flags[idx] = 1.0
        else:
            fp_flags[idx] = 1.0

        det_records.append({
            "img_id": img_id,
            "box": box,
            "score": d["score"],
            "is_tp": is_tp,
            "best_iou": best_iou,
        })

    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum(fp_flags)
    recalls = cum_tp / max(1, total_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1.0)

    ap, mrec, mpre_env = compute_ap_from_pr(recalls, precisions)
    return {
        "recalls": recalls,
        "precisions": precisions,
        "mrec": mrec,
        "mpre_env": mpre_env,
        "ap": ap,
        "cum_tp": cum_tp,
        "cum_fp": cum_fp,
        "total_gt": total_gt,
        "records": det_records,
    }


def verify_ap_lecture8():
    """Перевірка реалізації AP на еталонному прикладі з лекції 8 (n_gt=4, [TP, TP, FP, FP, TP] -> AP = 0.65)."""
    tp = np.array([1, 1, 0, 0, 1], dtype=np.float64)
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    rec = cum_tp / 4.0
    prec = cum_tp / (cum_tp + cum_fp)
    ap_val, _, _ = compute_ap_from_pr(rec, prec)
    assert abs(ap_val - 0.65) < 1e-9, f"Очікувалось AP=0.65, отримано {ap_val}"
    return ap_val


def trim_hog_context_margin(boxes: np.ndarray, pad_x: float = 0.16, pad_y: float = 0.10) -> np.ndarray:
    """
    Компенсує контекстні поля вікна Dalal--Triggs 64x128 (приблизно 16% по ширині та 10% по висоті з кожного боку),
    приводячи рамки детектора HOG до щільного формату еталонної розмітки INRIA Person.
    """
    if len(boxes) == 0:
        return boxes.copy()
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    return np.column_stack([
        boxes[:, 0] + pad_x * ws,
        boxes[:, 1] + pad_y * hs,
        boxes[:, 2] - pad_x * ws,
        boxes[:, 3] - pad_y * hs,
    ])


def run_part_e():
    print("=== ЧАСТИНА E: Детектор пішоходів HOG+SVM, NMS, крива Precision--Recall та AP ===")
    ap_check = verify_ap_lecture8()
    print(f"Перевірка власної реалізації AP (тест лекції 8): AP = {ap_check:.4f} (очікується 0.6500)")

    # Крок 11: Готовий детектор пішоходів HOG + лінійний SVM
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Крок 12: Відбір 30 кадрів із тестової вибірки INRIA Person (lab07/data/part_e/Test)
    ann_dir = PART_E_DIR / "Annotations"
    img_dir = PART_E_DIR / "JPEGImages"
    xml_files = sorted(ann_dir.glob("*.xml"))

    selected_items = []
    for xp in xml_files:
        fname, gt_boxes = parse_voc_xml(xp)
        if len(gt_boxes) == 0:
            continue
        ip = img_dir / fname
        if not ip.exists():
            continue
        selected_items.append((xp.stem, ip, gt_boxes))
        if len(selected_items) == 30:
            break

    total_gt_cnt = sum(len(x[2]) for x in selected_items)
    print(f"Відібрано {len(selected_items)} кадрів INRIA Person Test із {total_gt_cnt} еталонними пішоходами.")

    raw_trimmed_all = []
    nms_trimmed_all = []
    nms_untrimmed_all = []
    gt_by_image = {}
    img_cache = {}
    times_ms = []

    for img_id, ip, gt_boxes in selected_items:
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        gt_by_image[img_id] = gt_boxes

        t0 = time.perf_counter()
        # groupThreshold=0 вимикає внутрішнє групування OpenCV, щоб працював саме наш NMS (Крок 13)
        rects, weights = hog.detectMultiScale(
            img,
            hitThreshold=-0.20,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
            groupThreshold=0,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)

        if len(rects) > 0:
            boxes_raw = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects], dtype=np.float64)
            scores = np.array(weights, dtype=np.float64).ravel()
            boxes_trim = trim_hog_context_margin(boxes_raw, pad_x=0.16, pad_y=0.10)
        else:
            boxes_raw = np.empty((0, 4), dtype=np.float64)
            boxes_trim = np.empty((0, 4), dtype=np.float64)
            scores = np.empty((0,), dtype=np.float64)

        for tb, s in zip(boxes_trim, scores):
            raw_trimmed_all.append({"img_id": img_id, "box": tb, "score": float(s)})

        # Крок 13: Власний NMS (IoU > 0.5)
        nms_trim_b, nms_trim_s = non_max_suppression(boxes_trim, scores, iou_thresh=0.5)
        for b, s in zip(nms_trim_b, nms_trim_s):
            nms_trimmed_all.append({"img_id": img_id, "box": b, "score": float(s)})

        nms_raw_b, nms_raw_s = non_max_suppression(boxes_raw, scores, iou_thresh=0.5)
        for b, s in zip(nms_raw_b, nms_raw_s):
            nms_untrimmed_all.append({"img_id": img_id, "box": b, "score": float(s)})

        img_cache[img_id] = {
            "img": img,
            "gt": gt_boxes,
            "raw_n": len(boxes_raw),
        }

    # Оцінювання 4 режимів (Кроки 14--16)
    res_main = evaluate_detections_pr_ap(nms_trimmed_all, gt_by_image, iou_thresh=0.5, sort_by_conf=True)
    res_no_nms = evaluate_detections_pr_ap(raw_trimmed_all, gt_by_image, iou_thresh=0.5, sort_by_conf=True)
    res_unsorted = evaluate_detections_pr_ap(nms_trimmed_all, gt_by_image, iou_thresh=0.5, sort_by_conf=False)
    res_untrimmed = evaluate_detections_pr_ap(nms_untrimmed_all, gt_by_image, iou_thresh=0.5, sort_by_conf=True)

    f1_main = 2.0 * res_main["precisions"] * res_main["recalls"] / np.maximum(
        res_main["precisions"] + res_main["recalls"], 1e-9
    )
    best_idx = int(np.argmax(f1_main))
    best_f1 = float(f1_main[best_idx])
    best_p = float(res_main["precisions"][best_idx])
    best_r = float(res_main["recalls"][best_idx])
    best_thr = float(res_main["records"][best_idx]["score"])

    print(
        f"Сирих детекцій (groupThreshold=0): {len(raw_trimmed_all)} -> після NMS (IoU > 0.5): "
        f"{len(nms_trimmed_all)} (скорочення у {len(raw_trimmed_all)/max(1,len(nms_trimmed_all)):.1f}x)"
    )
    print(f"Середній час HOG+SVM на кадр: {np.mean(times_ms):.1f} ± {np.std(times_ms, ddof=1):.1f} мс")
    print(f"Основний конвеєр (корекція полів + NMS + сортування): AP = {res_main['ap']:.4f}, max F1 = {best_f1:.3f} (P={best_p:.3f}, R={best_r:.3f}, score>={best_thr:.2f})")
    print(f"Без NMS (8257 сирих рамок): AP = {res_no_nms['ap']:.4f}")
    print(f"Із NMS, але БЕЗ сортування за впевненістю: AP = {res_unsorted['ap']:.4f}")
    print(f"Без корекції контекстних полів вікна 64x128 + NMS: AP = {res_untrimmed['ap']:.4f}")

    # Збереження зведеної таблиці Частини E
    with open(RESULTS_DIR / "e_hog_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_dets", "tp_total", "fp_total", "max_recall", "max_f1", "ap"])
        for label, res_obj in [
            ("HOG + корекція полів + NMS + сортування", res_main),
            ("HOG + корекція полів без NMS (сирі рамки)", res_no_nms),
            ("HOG + корекція полів + NMS (без сортування)", res_unsorted),
            ("Сире вікно HOG 64x128 + NMS (без корекції полів)", res_untrimmed),
        ]:
            f1_arr = 2.0 * res_obj["precisions"] * res_obj["recalls"] / np.maximum(
                res_obj["precisions"] + res_obj["recalls"], 1e-9
            )
            writer.writerow([
                label,
                len(res_obj["records"]),
                int(res_obj["cum_tp"][-1]),
                int(res_obj["cum_fp"][-1]),
                f"{res_obj['recalls'][-1]:.3f}",
                f"{float(np.max(f1_arr)):.3f}",
                f"{res_obj['ap']:.4f}",
            ])

    # Порогова таблиця для основного конвеєра
    with open(RESULTS_DIR / "e_pr_thresholds.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score_thresh", "n_dets", "tp", "fp", "precision", "recall", "f1"])
        for thr in [1.50, 1.00, 0.70, 0.50, 0.25, 0.00, -0.20]:
            sub = [r for r in res_main["records"] if r["score"] >= thr]
            tp_c = sum(1 for r in sub if r["is_tp"])
            fp_c = sum(1 for r in sub if not r["is_tp"])
            pr_c = tp_c / max(1, tp_c + fp_c)
            rc_c = tp_c / max(1, res_main["total_gt"])
            f1_c = 2.0 * pr_c * rc_c / max(1e-9, pr_c + rc_c)
            writer.writerow([f"{thr:.2f}", len(sub), tp_c, fp_c, f"{pr_c:.3f}", f"{rc_c:.3f}", f"{f1_c:.3f}"])

    # Рисунок 5 (Частина E): Крива Precision--Recall з огинаючою AP та порівнянням режимів
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.9), dpi=200)

    r_grid = np.linspace(0.05, 1.0, 200)
    for f1_iso in [0.2, 0.4, 0.6, 0.8]:
        p_iso = (f1_iso * r_grid) / np.maximum(2.0 * r_grid - f1_iso, 1e-6)
        valid_iso = (2.0 * r_grid > f1_iso) & (p_iso <= 1.0)
        axes[0].plot(r_grid[valid_iso], p_iso[valid_iso], color="#cccccc", linestyle=":", linewidth=1.0)
        if np.any(valid_iso):
            idx_last = np.where(valid_iso)[0][-1]
            axes[0].text(r_grid[idx_last] - 0.07, p_iso[idx_last] + 0.015, f"F1={f1_iso:.1f}", fontsize=7.6, color="#777777")

    axes[0].plot(res_main["recalls"], res_main["precisions"], color="#4e79a7", linewidth=1.7, alpha=0.88,
                 label="Сира крива P(R) після NMS")
    axes[0].step(res_main["mrec"], res_main["mpre_env"], where="post", color="#d62728", linewidth=2.2,
                 label=f"Монотонна огинаюча (AP = {res_main['ap']:.3f})")
    axes[0].fill_between(res_main["mrec"], 0, res_main["mpre_env"], step="post", color="#d62728", alpha=0.14)
    axes[0].scatter([best_r], [best_p], color="#ffdf00", edgecolor="black", s=75, zorder=6,
                    label=f"Макс. F1 = {best_f1:.3f} (P={best_p:.2f}, R={best_r:.2f})")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("Повнота (Recall)", fontsize=9.8)
    axes[0].set_ylabel("Точність (Precision)", fontsize=9.8)
    axes[0].set_title(f"а) Крива Precision--Recall (30 кадрів INRIA, {res_main['total_gt']} GT)", fontsize=10.2, fontweight="bold")
    axes[0].grid(linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.2, loc="upper right")

    axes[1].step(res_main["mrec"], res_main["mpre_env"], where="post", color="#d62728", linewidth=2.3,
                 label=f"Корекція полів + NMS + сорт. (AP = {res_main['ap']:.3f})")
    axes[1].step(res_no_nms["mrec"], res_no_nms["mpre_env"], where="post", color="#f28e2b", linewidth=2.0, linestyle="--",
                 label=f"Без NMS, {len(raw_trimmed_all)} рамок (AP = {res_no_nms['ap']:.3f})")
    axes[1].step(res_unsorted["mrec"], res_unsorted["mpre_env"], where="post", color="#59a14f", linewidth=2.0, linestyle="-.",
                 label=f"NMS без сортування (AP = {res_unsorted['ap']:.3f})")
    axes[1].step(res_untrimmed["mrec"], res_untrimmed["mpre_env"], where="post", color="#76b7b2", linewidth=1.8, linestyle=":",
                 label=f"Сире вікно 64x128 + NMS (AP = {res_untrimmed['ap']:.3f})")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Повнота (Recall)", fontsize=9.8)
    axes[1].set_ylabel("Точність (огинаюча Precision)", fontsize=9.8)
    axes[1].set_title("б) Вплив NMS, сортування та контекстних полів вікна", fontsize=10.2, fontweight="bold")
    axes[1].grid(linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.1, loc="upper right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "e_pr_curve.png", bbox_inches="tight")
    plt.close(fig)

    # Рисунок 6 (Частина E): Приклади виявлення пішоходів (TP, FP, FN) при робочому порозі best_thr
    by_img_recs = {}
    for r in res_main["records"]:
        if r["score"] >= best_thr:
            by_img_recs.setdefault(r["img_id"], []).append(r)

    vis_img_ids = ["crop001501", "crop001512", "crop001573", "crop001639"]
    vis_img_ids = [iid for iid in vis_img_ids if iid in img_cache]
    if len(vis_img_ids) < 4:
        vis_img_ids = [x[0] for x in selected_items[:4]]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), dpi=170)
    for ax, img_id in zip(axes.ravel(), vis_img_ids):
        canvas = cv2.cvtColor(img_cache[img_id]["img"], cv2.COLOR_BGR2RGB).copy()
        gt_b = img_cache[img_id]["gt"]
        for gx1, gy1, gx2, gy2 in gt_b:
            cv2.rectangle(canvas, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 230, 80), 3)

        recs_i = by_img_recs.get(img_id, [])
        n_tp, n_fp = 0, 0
        for r in recs_i:
            x1, y1, x2, y2 = [int(round(v)) for v in r["box"]]
            if r["is_tp"]:
                col = (30, 144, 255)
                tag = f"TP {r['score']:.2f}"
                n_tp += 1
            else:
                col = (255, 45, 45)
                tag = f"FP {r['score']:.2f}"
                n_fp += 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3)
            cv2.putText(canvas, tag, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)

        ax.imshow(canvas)
        ax.set_title(
            f"{img_id}: GT={len(gt_b)}, TP={n_tp}, FP={n_fp}, FN={len(gt_b)-n_tp} (зелений=GT, синій=TP, червоний=FP)",
            fontsize=8.7, fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "e_hog_detections.png", bbox_inches="tight")
    plt.close(fig)
    print("Частину E успішно виконано!")


if __name__ == "__main__":
    run_part_e()
