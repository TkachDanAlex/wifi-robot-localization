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


def linear_ols_aoa(ap_xy: np.ndarray, global_phi: np.ndarray) -> np.ndarray:
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


def calc_stats(err_vec: np.ndarray) -> dict:
    return {
        "Замеров": len(err_vec),
        "Mean (м)": round(float(np.mean(err_vec)), 2),
        "Median (м)": round(float(np.median(err_vec)), 2),
        "RMSE (м)": round(float(np.sqrt(np.mean(err_vec**2))), 2),
        "95% (м)": round(float(np.percentile(err_vec, 95)), 2),
    }


def main():
    out_dir = "OLS_localisation"
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

    gt_xy = linear_ols_aoa(ap_xy, ap_yaw[:, None] + gnd_aoa)
    pred_phi = ap_yaw[:, None] + pred_aoa
    pred_lin_xy = linear_ols_aoa(ap_xy, pred_phi)

    err_lin = np.linalg.norm(pred_lin_xy - gt_xy, axis=1)

    summary_df = pd.DataFrame([
        {"Метод": "Линейный МНК (пересечение лучей)", **calc_stats(err_lin)},
    ])

    print("=== ЛИНЕЙНЫЙ МНК ПО УГЛАМ AoA (ВСЕ 4 AP) ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    np.savez_compressed(
        os.path.join(out_dir, "predictions.npz"),
        t=timestamps,
        gt_xy=gt_xy,
        pred_lin_xy=pred_lin_xy,
        ap_xy=ap_xy,
        ap_yaw=ap_yaw,
        aps=np.array(aps),
    )

    label_offsets = {
        "149": (4, 10),
        "151": (10, -4),
        "225": (8, 4),
        "227": (8, -6),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0))

    ax1.scatter(pred_lin_xy[::3, 0], pred_lin_xy[::3, 1], c="royalblue", s=3, alpha=0.14, zorder=2, label="Оценка позиции (линейный МНК)")
    ax1.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=1.2, alpha=0.85, zorder=4, label="Эталонная траектория")
    ax1.scatter(ap_xy[:, 0], ap_xy[:, 1], c="crimson", marker="^", s=130, edgecolor="black", zorder=6, label="Точки доступа (4 AP)")

    for idx, ap_id in enumerate(aps):
        ox, oy = label_offsets[ap_id]
        ax1.annotate(
            f"AP {ap_id}",
            (ap_xy[idx, 0], ap_xy[idx, 1]),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8.5,
            fontweight="bold",
            color="darkred",
        )

    ax1.set_xlim(-7.5, 15.0)
    ax1.set_ylim(-9.0, 5.0)
    ax1.set_title(
        f"Линейный МНК (пересечение лучей)\nMean={np.mean(err_lin):.2f}м, Median={np.median(err_lin):.2f}м, RMSE={np.sqrt(np.mean(err_lin**2)):.2f}м, P95={np.percentile(err_lin, 95):.2f}м",
        fontsize=10,
    )
    ax1.set_xlabel("X [м]")
    ax1.set_ylabel("Y [м]")
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    e_sorted = np.sort(err_lin)
    cdf = np.arange(1, len(e_sorted) + 1) / len(e_sorted)
    ax2.step(
        e_sorted,
        cdf,
        where="post",
        linewidth=2.2,
        color="royalblue",
        label=f"Линейный МНК (Mean={np.mean(err_lin):.2f}м, Med={np.median(err_lin):.2f}м, RMSE={np.sqrt(np.mean(err_lin**2)):.2f}м, P95={np.percentile(err_lin, 95):.2f}м)",
    )
    ax2.axhline(0.95, color="gray", linestyle=":", linewidth=1.1, label="Порог 95%")
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Распределение ошибок локализации (CDF)", fontsize=10.5)
    ax2.set_xlabel("Ошибка в плане XY [м]")
    ax2.set_ylabel("Доля замеров (CDF)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8.2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overview_linear.png"), dpi=200)

    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(len(gt_xy), size=20, replace=False))
    rel_t = timestamps[sample_idx] - timestamps[0]

    sample_rows = []
    for k, idx in enumerate(sample_idx, start=1):
        sample_rows.append({
            "№": k,
            "Индекс": int(idx),
            "Время (с)": round(float(rel_t[k - 1]), 1),
            "True X (м)": round(float(gt_xy[idx, 0]), 2),
            "True Y (м)": round(float(gt_xy[idx, 1]), 2),
            "Pred X (м)": round(float(pred_lin_xy[idx, 0]), 2),
            "Pred Y (м)": round(float(pred_lin_xy[idx, 1]), 2),
            "Ошибка (м)": round(float(err_lin[idx]), 2),
        })

    sample_df = pd.DataFrame(sample_rows)
    print("\n=== ПРОВЕРКА НА 20 СЛУЧАЙНЫХ ТОЧКАХ ТРАЕКТОРИИ ===")
    print(sample_df.to_string(index=False))
    sample_df.to_csv(os.path.join(out_dir, "random_20_points.csv"), index=False)

    fig_pts, ax_p = plt.subplots(figsize=(9.5, 6.2))
    ax_p.plot(gt_xy[:, 0], gt_xy[:, 1], color="lightgray", linewidth=1.1, zorder=1, label="Полная траектория")
    ax_p.scatter(ap_xy[:, 0], ap_xy[:, 1], c="crimson", marker="^", s=130, edgecolor="black", zorder=7, label="Точки доступа (4 AP)")

    for idx, ap_id in enumerate(aps):
        ox, oy = label_offsets[ap_id]
        ax_p.annotate(
            f"AP {ap_id}",
            (ap_xy[idx, 0], ap_xy[idx, 1]),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8.5,
            fontweight="bold",
            color="darkred",
        )

    for k, idx in enumerate(sample_idx, start=1):
        ax_p.plot(
            [gt_xy[idx, 0], pred_lin_xy[idx, 0]],
            [gt_xy[idx, 1], pred_lin_xy[idx, 1]],
            color="royalblue",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            zorder=4,
        )
        ax_p.annotate(
            str(k),
            (gt_xy[idx, 0], gt_xy[idx, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7.5,
            fontweight="bold",
            color="black",
            zorder=8,
        )

    ax_p.scatter(gt_xy[sample_idx, 0], gt_xy[sample_idx, 1], c="black", marker="o", s=45, zorder=6, label="Реальное положение (20 точек)")
    ax_p.scatter(pred_lin_xy[sample_idx, 0], pred_lin_xy[sample_idx, 1], c="royalblue", marker="x", s=65, linewidths=2.0, zorder=6, label="Оценка линейного МНК")

    ax_p.set_xlim(-7.5, 15.0)
    ax_p.set_ylim(-9.0, 5.0)
    ax_p.set_title(
        f"20 случайных точек: Линейный МНК\nMean={np.mean(err_lin[sample_idx]):.2f}м, Median={np.median(err_lin[sample_idx]):.2f}м",
        fontsize=10.5,
    )
    ax_p.set_xlabel("X [м]")
    ax_p.set_ylabel("Y [м]")
    ax_p.set_aspect("equal")
    ax_p.grid(True, linestyle=":", alpha=0.6)
    ax_p.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "random_20_points.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
