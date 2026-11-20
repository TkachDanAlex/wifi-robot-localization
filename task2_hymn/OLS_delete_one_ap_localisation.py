import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def find_file(rel_path: str, fallback_name: str) -> str:
    return rel_path if os.path.exists(rel_path) else fallback_name


def run_2d_lopo(
    ranges_clean: np.ndarray,
    true_3d: np.ndarray,
    wifi_clean: pd.DataFrame,
    active_pids: list,
    ap_xy: np.ndarray,
    ap_z: np.ndarray,
    keep_idx: np.ndarray,
) -> np.ndarray:
    preds = np.full((len(wifi_clean), 2), np.nan)
    sub_xy = ap_xy[keep_idx]
    sub_z = ap_z[keep_idx]
    x0 = np.mean(sub_xy, axis=0)

    for test_pid in active_pids:
        test_mask = (wifi_clean["point_id"] == test_pid).values
        train_mask = ~test_mask

        bias = np.nanmean(ranges_clean[train_mask] - true_3d[train_mask], axis=0)
        r_cal = ranges_clean[test_mask][:, keep_idx] - bias[None, keep_idx]
        z_test = wifi_clean.loc[test_mask, "Z_LOCAL_WIFI"].values
        d_2d = np.sqrt(np.maximum(0.0, r_cal**2 - (sub_z[None, :] - z_test[:, None]) ** 2))

        idxs = np.where(test_mask)[0]
        for k, idx in enumerate(idxs):
            ok = ~np.isnan(d_2d[k]) & (d_2d[k] > 0.0)
            if np.sum(ok) >= 3:
                res = least_squares(lambda xy: np.linalg.norm(sub_xy[ok] - xy, axis=1) - d_2d[k][ok], x0=x0)
                preds[idx] = res.x
    return preds


def build_ap_table(preds: np.ndarray, rx_xy: np.ndarray, zones: list, dropped_label: str) -> pd.DataFrame:
    rows = []
    for z_name, msk in zones:
        v = msk & ~np.isnan(preds[:, 0])
        err = np.linalg.norm(preds[v] - rx_xy[v], axis=1)
        rows.append({
            "Отключён": dropped_label,
            "Зона": z_name,
            "Решено": f"{np.sum(v)}/{np.sum(msk)} ({np.sum(v) / np.sum(msk) * 100:.1f}%)",
            "Mean (м)": round(float(np.mean(err)), 2),
            "Median (м)": round(float(np.median(err)), 2),
            "RMSE (м)": round(float(np.sqrt(np.mean(err**2))), 2),
            "95% (м)": round(float(np.percentile(err, 95)), 2),
        })
    return pd.DataFrame(rows)


def main():
    out_dir = "OLS_localisation"
    os.makedirs(out_dir, exist_ok=True)

    anchors = pd.read_csv(find_file("data/reference/csv/anchor_coordinates.csv", "anchor_coordinates.csv"))
    wifi = pd.read_csv(find_file("data/processed/csv/wifi.csv", "wifi.csv"))

    wifi_ap = anchors[anchors["point_id"].str.startswith("WIFI_")].sort_values("point_id").reset_index(drop=True)
    ap_names = list(wifi_ap["point_id"])
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
    rx_xy = rx_xyz[:, :2]
    true_3d = np.linalg.norm(rx_xyz[:, None, :] - ap_xyz[None, :, :], axis=2)

    active_pids = sorted(wifi_clean["point_id"].unique())
    all_mask = np.ones(len(wifi_clean), dtype=bool)
    in_mask = (~wifi_clean["point_id"].str.startswith("T")).values
    tr_mask = wifi_clean["point_id"].str.startswith("T").values
    zones = [("Все точки (47)", all_mask), ("Indoor (41)", in_mask), ("Transition (6)", tr_mask)]

    all_keep = np.arange(len(ap_names))
    base_preds = run_2d_lopo(ranges_clean, true_3d, wifi_clean, active_pids, ap_xy, ap_z, all_keep)
    base_df = build_ap_table(base_preds, rx_xy, zones, "Нет (все 6 AP)")
    print("=== ЭТАЛОН: ВСЕ 6 ТОЧЕК ДОСТУПА (2D МНК) ===")
    print(base_df.to_string(index=False))

    tables = [base_df]
    ap_preds_map = {}

    for drop_i, drop_name in enumerate(ap_names):
        keep_idx = np.array([j for j in range(len(ap_names)) if j != drop_i])
        p = run_2d_lopo(ranges_clean, true_3d, wifi_clean, active_pids, ap_xy, ap_z, keep_idx)
        ap_preds_map[drop_name] = p
        df_ap = build_ap_table(p, rx_xy, zones, drop_name)
        tables.append(df_ap)
        print(f"\n=== ОТКЛЮЧЁН {drop_name} (2D МНК) ===")
        print(df_ap.to_string(index=False))

    pd.concat(tables, ignore_index=True).to_csv(os.path.join(out_dir, "leave_one_ap_out_2d.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, msk, title, xmax in [
        (ax1, in_mask, "Внутри зала — Indoor (41 точка)", 12),
        (ax2, tr_mask, "За воротами — Transition (6 точек)", 35),
    ]:
        vb = msk & ~np.isnan(base_preds[:, 0])
        eb = np.sort(np.linalg.norm(base_preds[vb] - rx_xy[vb], axis=1))
        ax.step(eb, np.arange(1, len(eb) + 1) / len(eb), where="post", linewidth=2.5, color="black", label="Все 6 AP (эталон)")
        for drop_name, p in ap_preds_map.items():
            v = msk & ~np.isnan(p[:, 0])
            ea = np.sort(np.linalg.norm(p[v] - rx_xy[v], axis=1))
            ax.step(ea, np.arange(1, len(ea) + 1) / len(ea), where="post", linewidth=1.5, alpha=0.85, label=f"Без {drop_name}")
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=1.1)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Ошибка в плане XY (м)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8)

    ax1.set_ylabel("Доля замеров (CDF)")
    fig.suptitle("Leave-One-AP-Out (2D МНК): распределение ошибок при отключении роутера", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "leave_one_ap_out_2d_cdf.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
