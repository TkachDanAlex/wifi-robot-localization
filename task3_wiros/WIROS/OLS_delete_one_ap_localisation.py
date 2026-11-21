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
    valid = np.abs(det) > 1e-12
    safe_det = np.where(valid, det, 1.0)
    x = np.where(valid, (s_yy * s_xb - s_xy * s_yb) / safe_det, np.nan)
    y = np.where(valid, (-s_xy * s_xb + s_xx * s_yb) / safe_det, np.nan)
    return np.column_stack([x, y])


def nonlinear_ols_aoa(ap_xy: np.ndarray, global_phi: np.ndarray, n_iter: int = 30) -> np.ndarray:
    n_steps = global_phi.shape[1]
    p = np.tile(np.mean(ap_xy, axis=0), (n_steps, 1))
    lam = np.full(n_steps, 1e-3)

    def eval_residuals(pts: np.ndarray):
        dx = pts[None, :, 0] - ap_xy[:, 0:1]
        dy = pts[None, :, 1] - ap_xy[:, 1:2]
        res = wrap_angle(np.arctan2(dy, dx) - global_phi)
        return res, np.sum(res * res, axis=0)

    r, cost = eval_residuals(p)
    for _ in range(n_iter):
        dx = p[None, :, 0] - ap_xy[:, 0:1]
        dy = p[None, :, 1] - ap_xy[:, 1:2]
        n2 = np.maximum(dx * dx + dy * dy, 1e-8)

        jx = -dy / n2
        jy = dx / n2

        j11 = np.sum(jx * jx, axis=0) + lam
        j12 = np.sum(jx * jy, axis=0)
        j22 = np.sum(jy * jy, axis=0) + lam

        g1 = np.sum(jx * r, axis=0)
        g2 = np.sum(jy * r, axis=0)

        det = j11 * j22 - j12 * j12
        step_x = (j22 * g1 - j12 * g2) / det
        step_y = (-j12 * g1 + j11 * g2) / det

        p_cand = np.column_stack([p[:, 0] - step_x, p[:, 1] - step_y])
        r_cand, cost_cand = eval_residuals(p_cand)

        imp = cost_cand < cost
        p[imp] = p_cand[imp]
        r[:, imp] = r_cand[:, imp]
        cost[imp] = cost_cand[imp]
        lam = np.where(imp, np.maximum(lam * 0.5, 1e-7), np.minimum(lam * 5.0, 1e6))

    return p


def calc_stats(err_vec: np.ndarray) -> dict:
    valid_err = err_vec[~np.isnan(err_vec)]
    return {
        "Замеров": len(valid_err),
        "Mean (м)": round(float(np.mean(valid_err)), 2),
        "Median (м)": round(float(np.median(valid_err)), 2),
        "RMSE (м)": round(float(np.sqrt(np.mean(valid_err**2))), 2),
        "95% (м)": round(float(np.percentile(valid_err, 95)), 2),
    }


def main():
    out_dir = "OLS_leave_one_ap_out"
    os.makedirs(out_dir, exist_ok=True)

    data_dir = find_data_dir()
    aps = ["149", "151", "225", "227"]

    ap_xy = np.zeros((len(aps), 2))
    ap_yaw = np.zeros(len(aps))
    gnd_aoa = []
    pred_aoa = []

    for idx, ap_id in enumerate(aps):
        gnd_npz = np.load(os.path.join(data_dir, "gnd", f"gnd_{ap_id}.npz"))
        pred_npz = np.load(os.path.join(data_dir, "pred", f"{ap_id}-synced_aoa.npz"))

        ap_xy[idx] = gnd_npz["xy"]
        ap_yaw[idx] = float(gnd_npz["yaw"])
        gnd_aoa.append(gnd_npz["gnd"])
        pred_aoa.append(pred_npz["aoa"])

    gnd_aoa = np.asarray(gnd_aoa)
    pred_aoa = np.asarray(pred_aoa)

    gt_xy = linear_ols_aoa(ap_xy, ap_yaw[:, None] + gnd_aoa)
    pred_phi = ap_yaw[:, None] + pred_aoa

    configs = [("Все 4 AP", [0, 1, 2, 3])] + [
        (f"Без AP {aps[drop_idx]}", [i for i in range(len(aps)) if i != drop_idx])
        for drop_idx in range(len(aps))
    ]

    rows = []
    errors_lin = {}
    errors_nlin = {}

    for cfg_name, idxs in configs:
        sub_xy = ap_xy[idxs]
        sub_phi = pred_phi[idxs]

        est_lin = linear_ols_aoa(sub_xy, sub_phi)
        est_nlin = nonlinear_ols_aoa(sub_xy, sub_phi)

        e_lin = np.linalg.norm(est_lin - gt_xy, axis=1)
        e_nlin = np.linalg.norm(est_nlin - gt_xy, axis=1)

        errors_lin[cfg_name] = e_lin
        errors_nlin[cfg_name] = e_nlin

        rows.append({"Конфигурация": cfg_name, "Метод": "Линейный МНК", **calc_stats(e_lin)})
        rows.append({"Конфигурация": cfg_name, "Метод": "Нелинейный МНК", **calc_stats(e_nlin)})

    results_df = pd.DataFrame(rows)
    print("=== LEAVE-ONE-AP-OUT: СРАВНЕНИЕ ЛИНЕЙНОГО И НЕЛИНЕЙНОГО МНК ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(out_dir, "leave_one_ap_out.csv"), index=False)

    palette = {
        "Все 4 AP": ("black", "-", 2.5),
        "Без AP 149": ("tab:blue", "--", 1.8),
        "Без AP 151": ("tab:green", "-.", 2.0),
        "Без AP 225": ("tab:red", ":", 2.0),
        "Без AP 227": ("tab:purple", "--", 1.8),
    }

    fig_cdf, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)

    for ax, err_dict, title in [
        (ax1, errors_lin, "Линейный МНК (пересечение лучей)"),
        (ax2, errors_nlin, "Нелинейный МНК (угловые невязки)"),
    ]:
        for cfg_name, _ in configs:
            e_valid = err_dict[cfg_name]
            e_valid = e_valid[~np.isnan(e_valid)]
            e_sorted = np.sort(e_valid)
            cdf = np.arange(1, len(e_sorted) + 1) / len(e_sorted)
            color, ls, lw = palette[cfg_name]
            ax.step(
                e_sorted,
                cdf,
                where="post",
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=f"{cfg_name} (Med={np.median(e_valid):.2f}м, P95={np.percentile(e_valid, 95):.2f}м)",
            )

        ax.axhline(0.95, color="gray", linestyle=":", linewidth=1.1, label="Порог 95%")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Leave-One-AP-Out: {title}", fontsize=10.5)
        ax.set_xlabel("Ошибка в плане XY [м]")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8.2)

    ax1.set_ylabel("Доля замеров (CDF)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "leave_one_ap_out_cdf.png"), dpi=200)

    cfg_names = [c[0] for c in configs]
    x_pos = np.arange(len(cfg_names))
    width = 0.36

    fig_bar, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    metric_specs = [
        ("Median (м)", "Медианная ошибка (Median)"),
        ("Mean (м)", "Средняя ошибка (Mean)"),
        ("95% (м)", "95-й процентиль ошибки (P95)"),
    ]

    for ax, (col_name, title) in zip(axes, metric_specs):
        lin_vals = [
            results_df[(results_df["Конфигурация"] == c) & (results_df["Метод"] == "Линейный МНК")][col_name].values[0]
            for c in cfg_names
        ]
        nlin_vals = [
            results_df[(results_df["Конфигурация"] == c) & (results_df["Метод"] == "Нелинейный МНК")][col_name].values[0]
            for c in cfg_names
        ]

        bars1 = ax.bar(x_pos - width / 2, lin_vals, width, color="royalblue", label="Линейный МНК")
        bars2 = ax.bar(x_pos + width / 2, nlin_vals, width, color="darkorange", label="Нелинейный МНК")

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=7.8)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=7.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(cfg_names, rotation=18, ha="right", fontsize=8.8)
        ax.set_ylim(0, max(max(lin_vals), max(nlin_vals)) * 1.18)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("Ошибка [м]")
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8.2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "leave_one_ap_out_bars.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
