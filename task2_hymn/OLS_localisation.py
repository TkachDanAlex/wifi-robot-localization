import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import least_squares


def find_file(rel_path: str, fallback_name: str) -> str:
    return rel_path if os.path.exists(rel_path) else fallback_name


def localize_2d(
    r_3d: np.ndarray,
    rx_z: np.ndarray,
    ap_xy: np.ndarray,
    ap_z: np.ndarray,
    bias_3d: np.ndarray,
) -> np.ndarray:
    r_cal = r_3d - bias_3d[None, :]
    dz = ap_z[None, :] - rx_z[:, None]
    d_2d = np.sqrt(np.maximum(0.0, r_cal**2 - dz**2))

    x0 = np.mean(ap_xy, axis=0)
    preds = np.full((len(r_cal), 2), np.nan)
    for i in range(len(r_cal)):
        row_d = d_2d[i]
        ok = ~np.isnan(row_d) & (row_d > 0.0)
        if np.sum(ok) < 3:
            continue
        res = least_squares(lambda xy: np.linalg.norm(ap_xy[ok] - xy, axis=1) - row_d[ok], x0=x0)
        preds[i] = res.x
    return preds


def localize_3d(
    r_3d: np.ndarray,
    ap_xyz: np.ndarray,
    bias_3d: np.ndarray,
) -> np.ndarray:
    r_cal = r_3d - bias_3d[None, :]
    x0 = np.array([np.mean(ap_xyz[:, 0]), np.mean(ap_xyz[:, 1]), 2.5])
    preds = np.full((len(r_cal), 3), np.nan)
    for i in range(len(r_cal)):
        row_r = r_cal[i]
        ok = ~np.isnan(row_r) & (row_r > 0.0)
        if np.sum(ok) < 4:
            continue
        res = least_squares(lambda xyz: np.linalg.norm(ap_xyz[ok] - xyz, axis=1) - row_r[ok], x0=x0)
        preds[i] = res.x
    return preds


def calc_stats(err_vec: np.ndarray, total_count: int) -> dict:
    valid = err_vec[~np.isnan(err_vec)]
    n_ok = len(valid)
    return {
        "Решено замеров": f"{n_ok}/{total_count} ({n_ok / total_count * 100:.1f}%)",
        "Mean (м)": round(float(np.mean(valid)), 2),
        "Median (м)": round(float(np.median(valid)), 2),
        "RMSE (м)": round(float(np.sqrt(np.mean(valid**2))), 2),
        "95% (м)": round(float(np.percentile(valid, 95)), 2),
    }


def main():
    out_dir = "OLS_localisation"
    os.makedirs(out_dir, exist_ok=True)

    anchors = pd.read_csv(find_file("data/reference/csv/anchor_coordinates.csv", "anchor_coordinates.csv"))
    points = pd.read_csv(find_file("data/reference/csv/point_coordinates.csv", "point_coordinates.csv"))
    wifi = pd.read_csv(find_file("data/processed/csv/wifi.csv", "wifi.csv"))

    wifi_ap = anchors[anchors["point_id"].str.startswith("WIFI_")].sort_values("point_id").reset_index(drop=True)
    ap_xyz = wifi_ap[["X_LOCAL", "Y_LOCAL", "Z_LOCAL"]].values
    ap_xy = ap_xyz[:, :2]
    ap_z = ap_xyz[:, 2]

    order = [4, 1, 0, 2, 3, 5]
    ranges_raw = np.vstack(
        wifi["ranges"].apply(lambda s: np.array(ast.literal_eval(s.replace("nan", "None")), dtype=float)).values
    )
    ranges_3d = ranges_raw[:, order]

    has_data = np.sum(~np.isnan(ranges_3d), axis=1) > 0
    wifi_clean = wifi[has_data].reset_index(drop=True)
    ranges_clean = ranges_3d[has_data]

    rx_xyz = wifi_clean[["X_LOCAL_WIFI", "Y_LOCAL_WIFI", "Z_LOCAL_WIFI"]].values
    true_3d = np.linalg.norm(rx_xyz[:, None, :] - ap_xyz[None, :, :], axis=2)

    active_pids = sorted(wifi_clean["point_id"].unique())
    preds_2d = np.full((len(wifi_clean), 2), np.nan)
    preds_3d = np.full((len(wifi_clean), 3), np.nan)

    for test_pid in active_pids:
        test_mask = (wifi_clean["point_id"] == test_pid).values
        train_mask = ~test_mask

        bias = np.nanmean(ranges_clean[train_mask] - true_3d[train_mask], axis=0)

        r_test = ranges_clean[test_mask]
        z_test = wifi_clean.loc[test_mask, "Z_LOCAL_WIFI"].values

        preds_2d[test_mask] = localize_2d(r_test, z_test, ap_xy, ap_z, bias)
        preds_3d[test_mask] = localize_3d(r_test, ap_xyz, bias)

    in_mask = (~wifi_clean["point_id"].str.startswith("T")).values
    tr_mask = wifi_clean["point_id"].str.startswith("T").values
    zones = [
        ("Все точки (47)", np.ones(len(wifi_clean), dtype=bool)),
        ("Indoor (41)", in_mask),
        ("Transition (6)", tr_mask),
    ]

    table_rows = []
    for z_name, mask in zones:
        total_n = int(np.sum(mask))
        e_2d_xy = np.linalg.norm(preds_2d[mask] - rx_xyz[mask, :2], axis=1)
        e_3d_xy = np.linalg.norm(preds_3d[mask, :2] - rx_xyz[mask, :2], axis=1)
        e_3d_z = np.abs(preds_3d[mask, 2] - rx_xyz[mask, 2])
        e_3d_xyz = np.linalg.norm(preds_3d[mask] - rx_xyz[mask], axis=1)

        table_rows.append({"Метод": "2D МНК (ошибка в плане XY)", "Зона": z_name, **calc_stats(e_2d_xy, total_n)})
        table_rows.append({"Метод": "3D МНК (ошибка в плане XY)", "Зона": z_name, **calc_stats(e_3d_xy, total_n)})
        table_rows.append({"Метод": "3D МНК (ошибка по высоте Z)", "Зона": z_name, **calc_stats(e_3d_z, total_n)})
        table_rows.append({"Метод": "3D МНК (полная 3D-ошибка XYZ)", "Зона": z_name, **calc_stats(e_3d_xyz, total_n)})

    summary_df = pd.DataFrame(table_rows)
    print("\n=== ЕДИНАЯ СВОДНАЯ ТАБЛИЦА: 2D МНК ПРОТИВ 3D МНК ===")
    print(summary_df.to_string(index=False))

    metrics_path = os.path.join(out_dir, "metrics.csv")
    summary_df.to_csv(metrics_path, index=False)

    wifi_clean["pred2d_x"] = preds_2d[:, 0]
    wifi_clean["pred2d_y"] = preds_2d[:, 1]
    wifi_clean["pred3d_x"] = preds_3d[:, 0]
    wifi_clean["pred3d_y"] = preds_3d[:, 1]
    wifi_clean["pred3d_z"] = preds_3d[:, 2]

    pt_sum = wifi_clean.groupby("point_id").agg(
        true_x=("X_LOCAL_WIFI", "first"),
        true_y=("Y_LOCAL_WIFI", "first"),
        true_z=("Z_LOCAL_WIFI", "first"),
        p2d_x=("pred2d_x", "median"),
        p2d_y=("pred2d_y", "median"),
        p3d_x=("pred3d_x", "median"),
        p3d_y=("pred3d_y", "median"),
        p3d_z=("pred3d_z", "median"),
    ).reset_index()

    in_sum = pt_sum[~pt_sum["point_id"].str.startswith("T")]
    tr_sum = pt_sum[pt_sum["point_id"].str.startswith("T")]
    a04b4 = points[points["point_id"] == "A04B4"].iloc[0]

    fig, (ax2d, ax3d) = plt.subplots(1, 2, figsize=(18, 9.5), sharex=True, sharey=True)
    x_min, x_max = -12.71, 5.29
    y_min, y_max = -6.92, 37.08

    def draw_hall_and_truth(ax):
        hall_rect = patches.Rectangle((x_min, y_min), 18.0, 44.0, linewidth=2, edgecolor="black", facecolor="#f7f9fb", zorder=1)
        ax.add_patch(hall_rect)
        ax.plot([x_min, x_min], [-4.5, -1.5], color="#f7f9fb", linewidth=4.5, zorder=2)
        ax.plot([x_max, x_max], [-4.5, -1.5], color="#f7f9fb", linewidth=4.5, zorder=2)

        ax.scatter(ap_xy[:, 0], ap_xy[:, 1], c="crimson", marker="^", s=120, edgecolor="black", zorder=8, label="Wi-Fi роутеры (6 AP)")
        for _, r in wifi_ap.iterrows():
            ax.annotate(r["point_id"], (r["X_LOCAL"], r["Y_LOCAL"]), textcoords="offset points", xytext=(5, 4), fontsize=8, fontweight="bold", color="darkred")

        ax.scatter(in_sum["true_x"], in_sum["true_y"], c="black", marker="o", s=38, zorder=6, label="Истинные точки Indoor (41)")
        ax.scatter(tr_sum["true_x"], tr_sum["true_y"], c="white", marker="D", s=48, edgecolor="black", linewidth=1.2, zorder=6, label="Истинные точки Transition (6)")
        ax.scatter([a04b4["X_LOCAL_WIFI"]], [a04b4["Y_LOCAL_WIFI"]], c="gray", marker="s", s=40, zorder=5, label="A04B4 (нет данных)")

        for _, r in pt_sum.iterrows():
            if r["point_id"].startswith("T") or r["point_id"] in ["A13B2", "A13B6", "A01B1", "A01B6", "A07B3"]:
                ax.annotate(r["point_id"], (r["true_x"], r["true_y"]), textcoords="offset points", xytext=(4, -10), fontsize=7.5, fontweight="bold")

    draw_hall_and_truth(ax2d)
    ax2d.scatter(wifi_clean["pred2d_x"], wifi_clean["pred2d_y"], c="dodgerblue", s=4, alpha=0.12, zorder=2, label="Одиночные замеры (2D МНК)")
    for _, r in pt_sum.iterrows():
        ax2d.plot([r["true_x"], r["p2d_x"]], [r["true_y"], r["p2d_y"]], color="royalblue", linewidth=0.9, alpha=0.75, zorder=5)
    ax2d.scatter(pt_sum["p2d_x"], pt_sum["p2d_y"], c="royalblue", marker="x", s=55, linewidths=1.8, zorder=7, label="Предсказание 2D МНК (медиана)")
    ax2d.set_title("2D МНК (с известной высотой Z приёмника)\nIndoor: Mean=2.46м, RMSE=3.05м, P95=6.04м", fontsize=10.5)
    ax2d.set_xlabel("X_LOCAL [м]")
    ax2d.set_ylabel("Y_LOCAL [м]")
    ax2d.set_aspect("equal")
    ax2d.grid(True, linestyle=":", alpha=0.6)
    ax2d.legend(loc="upper left", fontsize=8, framealpha=0.92)

    draw_hall_and_truth(ax3d)
    ax3d.scatter(
        wifi_clean["pred3d_x"], wifi_clean["pred3d_y"],
        c=wifi_clean["pred3d_z"], cmap="viridis", vmin=1.5, vmax=4.8,
        s=4, alpha=0.16, zorder=2
    )
    for _, r in pt_sum.iterrows():
        ax3d.plot([r["true_x"], r["p3d_x"]], [r["true_y"], r["p3d_y"]], color="gray", linewidth=0.9, alpha=0.7, zorder=5)
    sc3d = ax3d.scatter(
        pt_sum["p3d_x"], pt_sum["p3d_y"],
        c=pt_sum["p3d_z"], cmap="viridis", vmin=1.5, vmax=4.8,
        marker="s", s=70, edgecolor="black", linewidth=0.9, zorder=7,
        label="Предсказание 3D МНК (цвет = высота Z)"
    )
    for _, r in pt_sum.iterrows():
        if r["p3d_z"] < 1.5:
            ax3d.annotate(f"z={r['p3d_z']:.1f}м", (r["p3d_x"], r["p3d_y"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color="indigo", fontweight="bold")

    ax3d.set_title("3D МНК (одновременный поиск X, Y, Z)\nIndoor XY: Mean=2.51м, RMSE=3.18м, P95=6.61м", fontsize=10.5)
    ax3d.set_xlabel("X_LOCAL [м]")
    ax3d.set_aspect("equal")
    ax3d.grid(True, linestyle=":", alpha=0.6)
    ax3d.legend(loc="upper left", fontsize=8, framealpha=0.92)

    cbar = plt.colorbar(sc3d, ax=ax3d, fraction=0.04, pad=0.02, shrink=0.82, extend="both")
    cbar.set_label("Предсказанная высота Z_LOCAL [м]\n(истина: 2.0–3.3 м; <1.5 м — выбросы вниз)", fontsize=9.5)

    plt.tight_layout()
    map_path = os.path.join(out_dir, "map.png")
    plt.savefig(map_path, dpi=200, bbox_inches="tight")

    fig_cdf, (ax_in, ax_tr) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, msk, title, xmax in [
        (ax_in, in_mask, "Внутри зала — Indoor (41 точка стояния)", 12),
        (ax_tr, tr_mask, "За воротами — Transition (6 точек стояния)", 35),
    ]:
        e2 = np.linalg.norm(preds_2d[msk] - rx_xyz[msk, :2], axis=1)
        e2 = np.sort(e2[~np.isnan(e2)])
        cdf2 = np.arange(1, len(e2) + 1) / len(e2)

        e3 = np.linalg.norm(preds_3d[msk, :2] - rx_xyz[msk, :2], axis=1)
        e3 = np.sort(e3[~np.isnan(e3)])
        cdf3 = np.arange(1, len(e3) + 1) / len(e3)

        ax.step(
            e2, cdf2, where="post", linewidth=2.2, color="royalblue",
            label=f"2D МНК (Mean={np.mean(e2):.2f}м, RMSE={np.sqrt(np.mean(e2**2)):.2f}м, P95={np.percentile(e2, 95):.2f}м)",
        )
        ax.step(
            e3, cdf3, where="post", linewidth=2.2, linestyle="--", color="darkorange",
            label=f"3D МНК (Mean={np.mean(e3):.2f}м, RMSE={np.sqrt(np.mean(e3**2)):.2f}м, P95={np.percentile(e3, 95):.2f}м)",
        )
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=1.1, label="Порог 95% замеров")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Ошибка в плане XY (м)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8.5)

    ax_in.set_ylabel("Доля одиночных замеров (CDF)")
    fig_cdf.suptitle("Сравнение распределения горизонтальных ошибок: 2D МНК против 3D МНК", fontsize=12)
    plt.tight_layout()
    cdf_path = os.path.join(out_dir, "cdf.png")
    plt.savefig(cdf_path, dpi=200)

    print(f"\nСохранено в папку {out_dir}/: metrics.csv, map.png, cdf.png")
    plt.show()


if __name__ == "__main__":
    main()
