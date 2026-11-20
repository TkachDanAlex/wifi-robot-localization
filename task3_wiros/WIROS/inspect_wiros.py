import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_data_dir() -> str:
    candidates = [
        "data/forward_localization",
        "forward_localization",
        "../../WIROS/wiros_processing_node/examples/forward_localization",
        "../WIROS/wiros_processing_node/examples/forward_localization",
        ".",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "gnd", "gnd_149.npz")):
            return path
    raise FileNotFoundError("Не найдена папка forward_localization с подпапками gnd/ и pred/")


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def triangulate_rays_linear(ap_xy: np.ndarray, global_phi: np.ndarray) -> np.ndarray:
    a_x = -np.sin(global_phi)
    a_y = np.cos(global_phi)
    b = a_x * ap_xy[:, 0:1] + a_y * ap_xy[:, 1:2]

    s_xx = np.sum(a_x * a_x, axis=0)
    s_xy = np.sum(a_x * a_y, axis=0)
    s_yy = np.sum(a_y * a_y, axis=0)
    s_xb = np.sum(a_x * b, axis=0)
    s_yb = np.sum(a_y * b, axis=0)

    det = s_xx * s_yy - s_xy * s_xy
    x = (s_yy * s_xb - s_xy * s_yb) / det
    y = (-s_xy * s_xb + s_xx * s_yb) / det
    return np.column_stack([x, y])


def main():
    out_dir = "wiros_inspect"
    os.makedirs(out_dir, exist_ok=True)

    data_dir = find_data_dir()
    aps = ["149", "151", "225", "227"]

    ap_xy = np.zeros((len(aps), 2))
    ap_yaw = np.zeros(len(aps))
    gnd_aoa = []
    pred_aoa = []
    timestamps = None

    for idx, ap_id in enumerate(aps):
        gnd_npz = np.load(os.path.join(data_dir, "gnd", f"gnd_{ap_id}.npz"))
        pred_npz = np.load(os.path.join(data_dir, "pred", f"{ap_id}-synced_aoa.npz"))

        ap_xy[idx] = gnd_npz["xy"]
        ap_yaw[idx] = float(gnd_npz["yaw"])
        gnd_aoa.append(gnd_npz["gnd"])
        pred_aoa.append(pred_npz["aoa"])
        if timestamps is None:
            timestamps = gnd_npz["t"]

    gnd_aoa = np.asarray(gnd_aoa)
    pred_aoa = np.asarray(pred_aoa)

    global_gnd_phi = ap_yaw[:, None] + gnd_aoa
    gt_xy = triangulate_rays_linear(ap_xy, global_gnd_phi)

    ang_check = np.arctan2(gt_xy[None, :, 1] - ap_xy[:, 1:2], gt_xy[None, :, 0] - ap_xy[:, 0:1])
    max_recon_err_deg = float(np.degrees(np.max(np.abs(wrap_angle(ang_check - global_gnd_phi)))))

    duration_sec = float(timestamps[-1] - timestamps[0])
    freq_hz = (len(timestamps) - 1) / duration_sec
    step_dist = np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)
    total_path_m = float(np.sum(step_dist))

    print(f"Отсчётов: {len(timestamps)} | Длительность: {duration_sec:.1f} с ({duration_sec / 60:.1f} мин) | Частота: {freq_hz:.1f} Гц")
    print(f"Длина маршрута робота: {total_path_m:.1f} м | Макс. невязка эталонных лучей: {max_recon_err_deg:.2e} град\n")

    rows = []
    err_deg_map = {}
    for idx, ap_id in enumerate(aps):
        err_deg = np.degrees(np.abs(wrap_angle(pred_aoa[idx] - gnd_aoa[idx])))
        err_deg_map[ap_id] = err_deg
        rows.append({
            "AP": ap_id,
            "X (м)": round(float(ap_xy[idx, 0]), 2),
            "Y (м)": round(float(ap_xy[idx, 1]), 2),
            "Yaw (град)": round(float(np.degrees(ap_yaw[idx])), 2),
            "Mean (град)": round(float(np.mean(err_deg)), 2),
            "Median (град)": round(float(np.median(err_deg)), 2),
            "RMSE (град)": round(float(np.sqrt(np.mean(err_deg**2))), 2),
            "95% (град)": round(float(np.percentile(err_deg, 95)), 2),
        })

    stats_df = pd.DataFrame(rows)
    print(stats_df.to_string(index=False))
    stats_df.to_csv(os.path.join(out_dir, "aoa_stats.csv"), index=False)

    np.savez_compressed(
        os.path.join(out_dir, "ground_truth_trajectory.npz"),
        t=timestamps,
        gt_xy=gt_xy,
        ap_xy=ap_xy,
        ap_yaw=ap_yaw,
        aps=np.array(aps),
    )

    fig, (ax_map, ax_cdf) = plt.subplots(1, 2, figsize=(15.5, 6.2))

    t_min = (timestamps - timestamps[0]) / 60.0
    sc = ax_map.scatter(gt_xy[::5, 0], gt_xy[::5, 1], c=t_min[::5], cmap="viridis", s=4, alpha=0.8, label="Эталонная траектория робота")
    ax_map.scatter(ap_xy[:, 0], ap_xy[:, 1], c="crimson", marker="^", s=140, edgecolor="black", zorder=6, label="Точки доступа WiROS (4 AP)")

    label_offsets = {
        "149": (4, 10),
        "151": (10, -4),
        "225": (8, 4),
        "227": (8, -6),
    }

    for idx, ap_id in enumerate(aps):
        dx = 1.6 * np.cos(ap_yaw[idx])
        dy = 1.6 * np.sin(ap_yaw[idx])
        ax_map.arrow(ap_xy[idx, 0], ap_xy[idx, 1], dx, dy, head_width=0.35, head_length=0.45, fc="darkred", ec="darkred", zorder=7)
        ox, oy = label_offsets[ap_id]
        ax_map.annotate(
            f"AP {ap_id}\n({ap_xy[idx, 0]:.1f}, {ap_xy[idx, 1]:.1f})",
            (ap_xy[idx, 0], ap_xy[idx, 1]),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8.5,
            fontweight="bold",
            color="darkred",
        )

    sample_idx = 15000
    ax_map.scatter([gt_xy[sample_idx, 0]], [gt_xy[sample_idx, 1]], c="black", marker="o", s=65, zorder=8, label=f"Пример позиции (t={t_min[sample_idx]:.1f} мин)")
    for idx in range(len(aps)):
        ax_map.plot([ap_xy[idx, 0], gt_xy[sample_idx, 0]], [ap_xy[idx, 1], gt_xy[sample_idx, 1]], linestyle="--", linewidth=1.0, color="gray", alpha=0.8, zorder=4)

    cbar = plt.colorbar(sc, ax=ax_map, fraction=0.042, pad=0.03, shrink=0.86)
    cbar.set_label("Время записи [мин]")
    ax_map.set_xlim(-7.2, 15.2)
    ax_map.set_ylim(-7.4, 4.3)
    ax_map.set_title("Полигон WiROS и восстановленная траектория робота")
    ax_map.set_xlabel("X [м]")
    ax_map.set_ylabel("Y [м]")
    ax_map.set_aspect("equal")
    ax_map.grid(True, linestyle=":", alpha=0.6)
    ax_map.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    for ap_id in aps:
        e_sorted = np.sort(err_deg_map[ap_id])
        cdf = np.arange(1, len(e_sorted) + 1) / len(e_sorted)
        ax_cdf.step(
            e_sorted,
            cdf,
            where="post",
            linewidth=2.0,
            label=f"AP {ap_id} (Med={np.median(e_sorted):.1f}°, P95={np.percentile(e_sorted, 95):.1f}°)",
        )

    ax_cdf.axhline(0.95, color="gray", linestyle=":", linewidth=1.1, label="Порог 95%")
    ax_cdf.set_xlim(0, 65)
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.set_title("Распределение ошибок измерения угла AoA (CDF)")
    ax_cdf.set_xlabel("Ошибка угла AoA [град]")
    ax_cdf.set_ylabel("Доля замеров")
    ax_cdf.grid(True, linestyle="--", alpha=0.6)
    ax_cdf.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overview.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
