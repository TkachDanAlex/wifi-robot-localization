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
    out_dir = "OLS_synthetic_noise"
    os.makedirs(out_dir, exist_ok=True)

    data_dir = find_data_dir()
    aps = ["149", "151", "225", "227"]

    ap_xy = np.zeros((len(aps), 2))
    ap_yaw = np.zeros(len(aps))
    gnd_aoa = []

    for idx, ap_id in enumerate(aps):
        gnd_npz = np.load(os.path.join(data_dir, "gnd", f"gnd_{ap_id}.npz"))
        ap_xy[idx] = gnd_npz["xy"]
        ap_yaw[idx] = float(gnd_npz["yaw"])
        gnd_aoa.append(gnd_npz["gnd"])

    gnd_aoa = np.asarray(gnd_aoa)
    gt_xy = linear_ols_aoa(ap_xy, ap_yaw[:, None] + gnd_aoa)

    noise_levels_deg = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    rng = np.random.default_rng(42)

    rows = []
    errors_lin = {}
    errors_nlin = {}

    for sigma_deg in noise_levels_deg:
        if sigma_deg == 0.0:
            noisy_aoa = gnd_aoa.copy()
        else:
            noise_rad = rng.normal(0.0, np.radians(sigma_deg), size=gnd_aoa.shape)
            noisy_aoa = wrap_angle(gnd_aoa + noise_rad)

        noisy_phi = ap_yaw[:, None] + noisy_aoa

        est_lin = linear_ols_aoa(ap_xy, noisy_phi)
        est_nlin = nonlinear_ols_aoa(ap_xy, noisy_phi)

        e_lin = np.linalg.norm(est_lin - gt_xy, axis=1)
        e_nlin = np.linalg.norm(est_nlin - gt_xy, axis=1)

        errors_lin[sigma_deg] = e_lin
        errors_nlin[sigma_deg] = e_nlin

        rows.append({
            "Эксперимент": "Синтетический шум AoA",
            "Шум σ (град)": sigma_deg,
            "Метод": "Линейный МНК",
            **calc_stats(e_lin),
        })
        rows.append({
            "Эксперимент": "Синтетический шум AoA",
            "Шум σ (град)": sigma_deg,
            "Метод": "Нелинейный МНК",
            **calc_stats(e_nlin),
        })

    results_df = pd.DataFrame(rows)
    print("=== [СИНТЕТИЧЕСКИЙ ЭКСПЕРИМЕНТ] ВЛИЯНИЕ УГЛОВОГО ШУМА N(0, σ²) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(out_dir, "synthetic_noise_metrics.csv"), index=False)

    fig_curves, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True)
    metric_specs = [
        ("Median (м)", "Медианная ошибка (Median)"),
        ("RMSE (м)", "Среднеквадратичная ошибка (RMSE)"),
        ("95% (м)", "95-й процентиль ошибки (P95)"),
    ]

    for ax, (col_name, title) in zip(axes, metric_specs):
        lin_df = results_df[results_df["Метод"] == "Линейный МНК"]
        nlin_df = results_df[results_df["Метод"] == "Нелинейный МНК"]

        ax.plot(lin_df["Шум σ (град)"], lin_df[col_name], marker="o", linewidth=2.2, color="royalblue", label="Линейный МНК (лучи)")
        ax.plot(nlin_df["Шум σ (град)"], nlin_df[col_name], marker="s", linestyle="--", linewidth=2.2, color="darkorange", label="Нелинейный МНК (углы)")

        for x_val, y_val in zip(lin_df["Шум σ (град)"], lin_df[col_name]):
            ax.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="royalblue")
        for x_val, y_val in zip(nlin_df["Шум σ (град)"], nlin_df[col_name]):
            ax.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8, color="darkorange")

        ax.set_title(f"[Синтетический шум]\n{title}", fontsize=10.5)
        ax.set_xlabel("СКО углового шума σ [град]")
        ax.set_ylabel("Ошибка позиционирования [м]")
        ax.set_xticks(noise_levels_deg)
        ax.set_ylim(0, max(lin_df[col_name].max(), nlin_df[col_name].max()) * 1.2 + 0.3)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "synthetic_noise_curves.png"), dpi=200)

    cdf_levels = [2.0, 5.0, 10.0, 15.0, 20.0]
    colors = ["tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple"]

    fig_cdf, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.0), sharey=True)
    for ax, err_dict, title in [
        (ax1, errors_lin, "Линейный МНК (пересечение лучей)"),
        (ax2, errors_nlin, "Нелинейный МНК (угловые невязки)"),
    ]:
        for sigma_deg, color in zip(cdf_levels, colors):
            e_valid = err_dict[sigma_deg]
            e_valid = e_valid[~np.isnan(e_valid)]
            e_sorted = np.sort(e_valid)
            cdf = np.arange(1, len(e_sorted) + 1) / len(e_sorted)
            ax.step(
                e_sorted,
                cdf,
                where="post",
                color=color,
                linewidth=2.0,
                label=f"σ = {sigma_deg:.0f}° (Med={np.median(e_valid):.2f}м, P95={np.percentile(e_valid, 95):.2f}м)",
            )

        ax.axhline(0.95, color="gray", linestyle=":", linewidth=1.1, label="Порог 95%")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"[Синтетический шум] {title}", fontsize=10.5)
        ax.set_xlabel("Ошибка в плане XY [м]")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8.5)

    ax1.set_ylabel("Доля замеров (CDF)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "synthetic_noise_cdf.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
