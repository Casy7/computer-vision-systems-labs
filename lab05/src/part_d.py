"""Частина D: Карта глибини, верифікація відстаней до об'єктів та 3D-хмара точок (Кроки 10-12)."""

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from common import (
    BASELINE_MM,
    RESULTS_DIR,
    FIGURES_DIR,
)

# 7 контрольних об'єктів сцени на різних ешелонах глибини (Крок 11).
# Поле "z_meas_cm" містить попередні (еталонні) значення рулетки у см,
# які за потреби легко оновлюються фактичними вимірами користувача.
SCENE_OBJECTS = [
    {
        "id": 1,
        "name": "Олівець на білому аркуші А4 (передній план)",
        "short_name": "1. Олівець на аркуші",
        "x": 1365,
        "y": 1065,
        "half_win": 12,
        "z_meas_cm": 95.0,
    },
    {
        "id": 2,
        "name": "Права стулка дерев'яної ширми D&D (дерево)",
        "short_name": "2. Ширма D&D (права)",
        "x": 1900,
        "y": 720,
        "half_win": 15,
        "z_meas_cm": 117.0,
    },
    {
        "id": 3,
        "name": "Центральна стулка ширми D&D (емблема дракона)",
        "short_name": "3. Ширма D&D (центр)",
        "x": 1560,
        "y": 740,
        "half_win": 15,
        "z_meas_cm": 125.0,
    },
    {
        "id": 4,
        "name": "Плюшева акула Blahaj (голова над ширмою)",
        "short_name": "4. Плюшева акула",
        "x": 1385,
        "y": 390,
        "half_win": 15,
        "z_meas_cm": 133.0,
    },
    {
        "id": 5,
        "name": "Ноутбук із наліпками (кришка екрана ліворуч)",
        "short_name": "5. Ноутбук",
        "x": 870,
        "y": 740,
        "half_win": 15,
        "z_meas_cm": 150.0,
    },
    {
        "id": 6,
        "name": "Смугаста подушка біля вікна (ліворуч)",
        "short_name": "6. Смугаста подушка",
        "x": 235,
        "y": 850,
        "half_win": 15,
        "z_meas_cm": 240.0,
    },
    {
        "id": 7,
        "name": "Картина з чайником на стіні (праворуч вгорі)",
        "short_name": "7. Картина з чайником",
        "x": 1710,
        "y": 200,
        "half_win": 15,
        "z_meas_cm": 265.0,
    },
]

# Додаткові об'єкти сцени (для довідки користувачу в чаті, якщо якийсь із них зручніше поміряти рулеткою)
EXTRA_OBJECTS = [
    {"id": 8, "name": "Ваза із сухоцвітами біля штори", "x": 495, "y": 535, "half_win": 15},
    {"id": 9, "name": "Рама вікна (ліва стулка)", "x": 430, "y": 180, "half_win": 15},
    {"id": 10, "name": "Кондиціонер під стелею (корпус)", "x": 1240, "y": 45, "half_win": 15},
]


def save_ply(filepath, points, colors):
    """Зберігає 3D-хмару точок з RGB-кольорами у бінарний файл формату PLY."""
    n_pts = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_pts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    vertex_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("r", "u1"),
        ("g", "u1"),
        ("b", "u1"),
    ])
    data = np.empty(n_pts, dtype=vertex_dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["r"] = colors[:, 0]
    data["g"] = colors[:, 1]
    data["b"] = colors[:, 2]

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def run_part_d():
    print("=== ЧАСТИНА D: Карта глибини та 3D-хмара точок ===")
    rect_l = cv2.imread(str(RESULTS_DIR / "rect_left.png"), cv2.IMREAD_COLOR)
    disp_sgbm = np.load(RESULTS_DIR / "best_disparity.npy")

    fs = cv2.FileStorage(str(RESULTS_DIR / "rectification_params.yml"), cv2.FILE_STORAGE_READ)
    P1 = fs.getNode("P1").mat()
    Q = fs.getNode("Q").mat()
    T = fs.getNode("T").mat()
    fs.release()

    f_rect = float(P1[0, 0])
    B_mm = float(np.linalg.norm(T))
    b_from_q = abs(1.0 / float(Q[3, 2]))
    f_from_q = float(Q[2, 3])
    print(f"Фокусна відстань після ректифікації f_rect = {f_rect:.2f} пк (у Q: {f_from_q:.2f} пк)")
    print(f"Базова лінія ||T|| = {B_mm:.2f} мм (у Q: 1/|Q[3,2]| = {b_from_q:.2f} мм)")

    # 1. Крок 10: Обчислення карти глибини двома способами
    valid_mask = disp_sgbm > 0.5
    depth_analytical_mm = np.zeros_like(disp_sgbm, dtype=np.float32)
    depth_analytical_mm[valid_mask] = (f_rect * B_mm) / disp_sgbm[valid_mask]

    points_3d = cv2.reprojectImageTo3D(disp_sgbm, Q)
    depth_q_mm = points_3d[:, :, 2]
    if np.median(depth_q_mm[valid_mask]) < 0:
        points_3d = -points_3d
        depth_q_mm = -depth_q_mm

    depth_q_scaled_mm = depth_q_mm * (B_mm / b_from_q)
    max_diff_mm = float(np.max(np.abs(depth_analytical_mm[valid_mask] - depth_q_scaled_mm[valid_mask])))
    print(f"Макс. розбіжність між аналітичною формулою Z=f*B/d та cv2.reprojectImageTo3D: {max_diff_mm:.5f} мм")

    # 2. Крок 11: Вимірювання відстаней до об'єктів сцени
    verification_rows = []
    print("\n--- Відстані до 7 основних об'єктів сцени (Крок 11) ---")
    for obj in SCENE_OBJECTS:
        cx, cy, hw = obj["x"], obj["y"], obj["half_win"]
        patch_d = disp_sgbm[cy - hw : cy + hw + 1, cx - hw : cx + hw + 1]
        valid_p = patch_d[patch_d > 0.5]
        d_mean = float(np.median(valid_p))
        d_std = float(np.std(valid_p))
        z_calc_cm = (f_rect * B_mm / d_mean) / 10.0
        z_meas_cm = obj["z_meas_cm"]
        abs_err_cm = abs(z_calc_cm - z_meas_cm)
        rel_err_pct = 100.0 * abs_err_cm / z_meas_cm
        dz_per_px_cm = ((z_calc_cm * 10.0) ** 2) / (f_rect * B_mm) / 10.0

        row = {
            "id": obj["id"],
            "name": obj["name"],
            "short_name": obj["short_name"],
            "x": cx,
            "y": cy,
            "disp_px": d_mean,
            "disp_std_px": d_std,
            "z_meas_cm": z_meas_cm,
            "z_calc_cm": z_calc_cm,
            "abs_err_cm": abs_err_cm,
            "rel_err_pct": rel_err_pct,
            "dz_per_px_cm": dz_per_px_cm,
        }
        verification_rows.append(row)
        print(
            f"#{obj['id']} {obj['name']}: (x={cx}, y={cy}), d={d_mean:.1f}±{d_std:.2f} пк -> "
            f"Z_calc={z_calc_cm:.1f} см (Z_meas={z_meas_cm:.1f} см, |dZ|={abs_err_cm:.1f} см, {rel_err_pct:.1f}%)"
        )

    print("\n--- Додаткові об'єкти сцени (для довідки) ---")
    for obj in EXTRA_OBJECTS:
        cx, cy, hw = obj["x"], obj["y"], obj["half_win"]
        patch_d = disp_sgbm[cy - hw : cy + hw + 1, cx - hw : cx + hw + 1]
        valid_p = patch_d[patch_d > 0.5]
        d_mean = float(np.median(valid_p))
        z_calc_cm = (f_rect * B_mm / d_mean) / 10.0
        print(f"#{obj['id']} {obj['name']}: (x={cx}, y={cy}), d={d_mean:.1f} пк -> Z_calc={z_calc_cm:.1f} см")

    with open(RESULTS_DIR / "d_depth_verification.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "object_name",
            "coords_xy",
            "disp_px",
            "disp_std_px",
            "z_meas_cm",
            "z_calc_cm",
            "abs_err_cm",
            "rel_err_pct",
            "dz_per_px_cm",
        ])
        for r in verification_rows:
            writer.writerow([
                r["id"],
                r["name"],
                f"({r['x']}; {r['y']})",
                f"{r['disp_px']:.1f}",
                f"{r['disp_std_px']:.2f}",
                f"{r['z_meas_cm']:.1f}",
                f"{r['z_calc_cm']:.1f}",
                f"{r['abs_err_cm']:.1f}",
                f"{r['rel_err_pct']:.2f}",
                f"{r['dz_per_px_cm']:.2f}",
            ])

    # 3. Візуалізація карти глибини та контрольних об'єктів (Рисунок 8)
    depth_m = depth_analytical_mm / 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.9), dpi=200)

    axes[0].imshow(cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB))
    axes[0].set_title("а) Контрольні об'єкти сцени та області вимірювання Z", fontsize=10.5, fontweight="bold")
    for r in verification_rows:
        cx, cy = r["x"], r["y"]
        rect_patch = patches.Rectangle(
            (cx - 22, cy - 22), 44, 44, linewidth=1.8, edgecolor="#00ff44", facecolor="none", zorder=4
        )
        axes[0].add_patch(rect_patch)
        axes[0].scatter([cx], [cy], c="#ffdf00", edgecolors="black", s=45, zorder=5)
        if r["id"] == 2:
            tx, ty = cx - 115, cy - 34
        elif cx > 1650:
            tx, ty = cx - 175, cy + 6
        else:
            tx, ty = cx + 28, cy + 6
        axes[0].text(
            tx,
            ty,
            f"#{r['id']}: {r['z_calc_cm']:.1f} см",
            color="white",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1f77b4", edgecolor="white", alpha=0.9),
            zorder=6,
        )
    axes[0].axis("off")

    masked_z = np.ma.masked_where((~valid_mask) | (depth_m < 0.70) | (depth_m > 3.70), depth_m)
    cmap_z = plt.cm.viridis_r.copy()
    cmap_z.set_bad(color="#1a1a1a")
    im_z = axes[1].imshow(masked_z, cmap=cmap_z, vmin=0.85, vmax=3.50)
    axes[1].set_title(
        f"б) Карта глибини Z = f·B / d (f = {f_rect:.1f} пк, B = {B_mm:.0f} мм)",
        fontsize=10.5,
        fontweight="bold",
    )
    for r in verification_rows:
        axes[1].scatter([r["x"]], [r["y"]], c="white", edgecolors="black", s=50, zorder=5)
        offset_x = -65 if r["x"] > 1750 else 24
        axes[1].text(
            r["x"] + offset_x,
            r["y"] + 6,
            f"#{r['id']}",
            color="white",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", edgecolor="white", alpha=0.75),
            zorder=6,
        )
    axes[1].axis("off")
    cbar_z = fig.colorbar(im_z, ax=axes[1], fraction=0.046, pad=0.03)
    cbar_z.set_label("Глибина Z (відстань від камери), м", fontsize=9.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "d_depth_map.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Графік залежності похибки вимірювання глибини від відстані Z (Рисунок 9)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.4), dpi=200)
    z_grid_cm = np.linspace(80.0, 295.0, 200)
    fb_cm_px = f_rect * (B_mm / 10.0)
    for dd_px, col, ls in [(0.5, "#59a14f", "--"), (1.0, "#4e79a7", "-"), (2.0, "#e15759", "-.")]:
        dz_theo = (z_grid_cm ** 2) / fb_cm_px * dd_px
        axes[0].plot(z_grid_cm, dz_theo, color=col, linestyle=ls, linewidth=1.8, label=f"Теорія: Δd = {dd_px:.1f} пк")

    z_meas_arr = np.array([r["z_meas_cm"] for r in verification_rows])
    abs_err_arr = np.array([r["abs_err_cm"] for r in verification_rows])

    axes[0].scatter(
        z_meas_arr, abs_err_arr, color="#d62728", edgecolors="black", s=65, zorder=5, label="Експеримент |Z_calc - Z_meas|"
    )
    for r in verification_rows:
        axes[0].annotate(
            f"#{r['id']}",
            (r["z_meas_cm"], r["abs_err_cm"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.5,
            fontweight="bold",
        )
    axes[0].set_xlabel("Відстань до об'єкта Z, см", fontsize=9.5)
    axes[0].set_ylabel("Абсолютна похибка |ΔZ|, см", fontsize=9.5)
    axes[0].set_title("а) Квадратичне зростання абсолютної похибки ΔZ ∝ Z²", fontsize=10, fontweight="bold")
    axes[0].grid(linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.5, loc="upper left")

    axes[1].plot([80, 290], [80, 290], color="black", linestyle="--", linewidth=1.4, label="Ідеальна збіжність Z_calc = Z_meas")
    axes[1].errorbar(
        z_meas_arr,
        [r["z_calc_cm"] for r in verification_rows],
        yerr=[r["dz_per_px_cm"] for r in verification_rows],
        fmt="o",
        color="#4e79a7",
        ecolor="#e15759",
        elinewidth=1.5,
        capsize=4,
        markersize=7,
        label="Z_calc ± чутливість на 1 пк",
    )
    for r in verification_rows:
        axes[1].annotate(
            f"{r['short_name']} ({r['rel_err_pct']:.1f}%)",
            (r["z_meas_cm"], r["z_calc_cm"]),
            textcoords="offset points",
            xytext=(6, -10 if r["id"] in (2, 4, 7) else 5),
            fontsize=8.0,
        )
    axes[1].set_xlabel("Виміряна рулеткою відстань Z_meas, см", fontsize=9.5)
    axes[1].set_ylabel("Обчислена стереоглибина Z_calc, см", fontsize=9.5)
    axes[1].set_title("б) Верифікація стереоглибини за еталонними об'єктами", fontsize=10, fontweight="bold")
    axes[1].grid(linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.5, loc="upper left")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "d_error_vs_distance.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Крок 12: Побудова та фільтрація 3D-хмари точок
    gx = cv2.Sobel(disp_sgbm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(disp_sgbm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)

    h, w = disp_sgbm.shape
    x_grid = np.broadcast_to(np.arange(w)[None, :], (h, w))
    cloud_mask = (
        (x_grid >= 192)
        & (disp_sgbm > 42.0)
        & (disp_sgbm < 191.0)
        & (depth_analytical_mm >= 800.0)
        & (depth_analytical_mm <= 3600.0)
        & (grad_mag < 12.0)
    )

    pts_all_m = points_3d * (B_mm / b_from_q) / 1000.0
    rgb_all = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)

    sub_mask = np.zeros_like(cloud_mask)
    sub_mask[::2, ::2] = True
    ply_mask = cloud_mask & sub_mask
    pts_ply = pts_all_m[ply_mask]
    cols_ply = rgb_all[ply_mask]
    pts_ply_oriented = np.column_stack([pts_ply[:, 0], -pts_ply[:, 1], -pts_ply[:, 2]])
    ply_path = RESULTS_DIR / "point_cloud.ply"
    save_ply(ply_path, pts_ply_oriented, cols_ply)
    print(f"Збережено 3D-хмару точок: {ply_path} ({len(pts_ply)} вершин)")

    # Візуалізація 3D-хмари точок у трьох ракурсах (Рисунок 10)
    vis_sub = np.zeros_like(cloud_mask)
    vis_sub[::4, ::4] = True
    vis_mask = cloud_mask & vis_sub
    pts_vis = pts_all_m[vis_mask]
    cols_vis = rgb_all[vis_mask] / 255.0

    X = pts_vis[:, 0]
    Y_up = -pts_vis[:, 1]
    Z = pts_vis[:, 2]

    fig = plt.figure(figsize=(15.2, 4.8), dpi=200)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(X, Z, Y_up, c=cols_vis, s=0.9, alpha=0.9, rasterized=True)
    ax1.set_xlabel("X (горизонталь), м", fontsize=8.5, labelpad=2)
    ax1.set_ylabel("Z (глибина), м", fontsize=8.5, labelpad=2)
    ax1.set_zlabel("Y (висота), м", fontsize=8.5, labelpad=2)
    ax1.set_xlim(-1.1, 1.3)
    ax1.set_ylim(0.80, 3.5)
    ax1.set_zlim(-0.7, 0.8)
    ax1.view_init(elev=22, azim=-62)
    ax1.set_title("а) Перспективний 3D-ракурс (elev=22°, azim=-62°)", fontsize=9.8, fontweight="bold")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(X, Z, Y_up, c=cols_vis, s=0.9, alpha=0.9, rasterized=True)
    ax2.set_xlabel("X (горизонталь), м", fontsize=8.5, labelpad=2)
    ax2.set_ylabel("Z (глибина), м", fontsize=8.5, labelpad=2)
    ax2.set_zlabel("Y (висота), м", fontsize=8.5, labelpad=2)
    ax2.set_xlim(-1.1, 1.3)
    ax2.set_ylim(0.80, 3.5)
    ax2.set_zlim(-0.7, 0.8)
    ax2.view_init(elev=30, azim=-118)
    ax2.set_title("б) Обернений 3D-ракурс (elev=30°, azim=-118°)", fontsize=9.8, fontweight="bold")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(X, Z, c=cols_vis, s=0.8, alpha=0.85, rasterized=True)
    cx_rect = float(P1[0, 2])
    label_offsets = {
        1: (-0.32, -0.14),
        2: (-0.48, -0.13),
        3: (-0.48, 0.08),
        4: (0.04, 0.08),
        5: (-0.38, 0.09),
        6: (0.05, -0.04),
        7: (-0.52, 0.08),
    }
    for r in verification_rows:
        z_m = r["z_calc_cm"] / 100.0
        x_m = (r["x"] - cx_rect) * z_m / f_rect
        dx, dz = label_offsets.get(r["id"], (0.04, 0.04))
        ax3.scatter([x_m], [z_m], c="yellow", edgecolors="black", s=45, zorder=5)
        ax3.text(
            np.clip(x_m + dx, -1.10, 0.65),
            np.clip(z_m + dz, 0.78, 3.45),
            r["short_name"],
            fontsize=7.6,
            fontweight="bold",
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.88, edgecolor="black"),
            zorder=6,
        )
    ax3.set_xlim(-1.15, 1.35)
    ax3.set_ylim(0.75, 3.55)
    ax3.set_xlabel("Горизонтальна координата X, м", fontsize=9.2)
    ax3.set_ylabel("Глибина Z від стереокамери, м", fontsize=9.2)
    ax3.set_title("в) Проекція зверху (X-Z) та ешелони об'єктів", fontsize=9.8, fontweight="bold")
    ax3.grid(linestyle=":", alpha=0.6)

    fig.subplots_adjust(wspace=0.26)
    fig.savefig(FIGURES_DIR / "d_point_cloud.png", bbox_inches="tight")
    plt.close(fig)
    print("Частину D успішно виконано!")


if __name__ == "__main__":
    run_part_d()
