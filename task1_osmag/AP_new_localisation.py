import os
import csv
import random
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from utils.configuration import ConfigurationManager
from utils.building_constraints import find_largest_polygon
from core.data_loader import DataLoader
from core.signal_model import SignalModelOptimizer
from core.fingerprint_point import WiFiFingerprintLocalizer
from io_layer.osm_parser import OsmDataParser, APMapParser, FingerprintParser
from algorithms.point_estimator import PointEstimator


def get_default_params():
    sig = inspect.signature(WiFiFingerprintLocalizer.__init__).parameters
    r0 = sig["rssi_0"].default if "rssi_0" in sig and sig["rssi_0"].default is not inspect.Parameter.empty else -30.0
    n_val = sig["n"].default if "n" in sig and sig["n"].default is not inspect.Parameter.empty else 2.0
    w_val = sig["ave_val"].default if "ave_val" in sig and sig["ave_val"].default is not inspect.Parameter.empty else -15.0
    return float(r0), float(n_val), float(w_val)


def calc_metrics(errors, total_count):
    valid = [float(e) for e in errors if e is not None and not np.isnan(e)]
    if not valid:
        return {"mean": np.nan, "median": np.nan, "rmse": np.nan, "p95": np.nan, "rate": 0.0, "ok": 0}
    arr = np.array(valid, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "p95": float(np.percentile(arr, 95)),
        "rate": float(len(arr) / total_count * 100.0),
        "ok": len(arr),
    }


def run_ap(ap_positions, edges, polygons, clusters, test_data, rssi_0, n, wall, use_constraints=True, iters=5, seed=10):
    random.seed(seed)
    np.random.seed(seed)
    localizer = WiFiFingerprintLocalizer(ap_positions, edges, rssi_0=rssi_0, n=n, ave_val=wall)

    orig_init = PointEstimator.__init__
    orig_est = PointEstimator.estimate_point

    def custom_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if not use_constraints:
            self.use_polygon_constraints = False
            self.constraint_polygons = []
            self.min_lon, self.max_lon = -180.0, 180.0
            self.min_lat, self.max_lat = -90.0, 90.0

    def custom_est(self, *args, **kwargs):
        if not use_constraints:
            kwargs["use_constrained_optimization"] = False
            if len(args) >= 3:
                args = (args[0], args[1], False) + args[3:]
        return orig_est(self, *args, **kwargs)

    try:
        PointEstimator.__init__ = custom_init
        PointEstimator.estimate_point = custom_est
        return localizer.localize_fingerprint_points(test_data, polygons, clusters, iters)
    finally:
        PointEstimator.__init__ = orig_init
        PointEstimator.estimate_point = orig_est


def main():
    out_dir = "AP_localisation"
    os.makedirs(out_dir, exist_ok=True)

    cfg = ConfigurationManager("./config/ap_localization_config.yaml")
    loader = DataLoader(cfg)
    ap_to_pos, ap_level, target_ap, _, _, _, polygon_edges, ap_learn_flag = loader.load_osm_data()

    sig_opt = SignalModelOptimizer(cfg)
    rssi_0_train, n_train, wall_train = sig_opt.optimize_propagation_model(
        target_ap, ap_to_pos, polygon_edges, ap_level, ap_learn_flag
    )
    rssi_0_def, n_def, wall_def = get_default_params()

    print(f"Train параметры:    RSSI_0={rssi_0_train:.4f}, n={n_train:.4f}, wall={wall_train:.4f}")
    print(f"Зашитые параметры:  RSSI_0={rssi_0_def:.4f}, n={n_def:.4f}, wall={wall_def:.4f}")

    ap_positions = APMapParser("./map/AP_MAP.osm").parse_ap_map()
    test_fp_data = FingerprintParser("./map/Non-FingerprintedAreas.osm").parse_fingerprint_data()
    total_test = len(test_fp_data)

    map_file = "./map/base_map.osm" if os.path.exists("./map/base_map.osm") else "./map/wifi_data.osm"
    poly_parser = OsmDataParser(map_file)
    poly_parser.parse()
    poly_parser.extract_polygon_edges()
    _, _, _, _, _, building_polygons, building_edges, _ = poly_parser.get_data()
    building_clusters = find_largest_polygon(building_polygons)

    param_sets = [
        ("Обученные (train)", rssi_0_train, n_train, wall_train),
        ("Зашитые (default)", rssi_0_def, n_def, wall_def),
    ]

    csv_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))

    print(f"\n{'Набор':<18} | {'Режим':<28} | {'Mean (м)':<8} | {'Median (м)':<10} | {'RMSE (м)':<8} | {'95% (м)':<8} | {'Решений':<12}")
    print("-" * 108)

    for row_idx, (set_name, r0, n_val, w_val) in enumerate(param_sets):
        ax_hist, ax_map, ax_cdf = axes[row_idx]

        iter_range = list(range(1, 16))
        iter_med, iter_rmse, iter_p95 = [], [], []
        for it in iter_range:
            res_it = run_ap(ap_positions, building_edges, building_polygons, building_clusters, test_fp_data, r0, n_val, w_val, True, it)
            m_it = calc_metrics([r["error_meters"] for r in res_it.values()], total_test)
            iter_med.append(m_it["median"])
            iter_rmse.append(m_it["rmse"])
            iter_p95.append(m_it["p95"])

        best_idx = int(np.nanargmin(iter_med))
        best_it = iter_range[best_idx]
        best_med = iter_med[best_idx]

        bars = ax_hist.bar(iter_range, iter_med, color="steelblue", alpha=0.75, width=0.65, label="Median (м)")
        bars[best_idx].set_color("crimson")
        ax_hist.plot(iter_range, iter_rmse, color="darkorange", marker="o", linewidth=1.5, linestyle="--", label="RMSE (м)")
        ax_hist.plot(iter_range, iter_p95, color="purple", marker="s", linewidth=1.5, linestyle="-.", label="95% (м)")
        ax_hist.set_title(f"{set_name}: ошибка по итерациям\n(мин. Median = {best_med:.2f} м при iter={best_it})", fontsize=10)
        ax_hist.set_xlabel("iter (число итераций)")
        ax_hist.set_ylabel("Ошибка (м)")
        ax_hist.set_xticks(iter_range)
        ax_hist.grid(True, linestyle=":", alpha=0.6)
        ax_hist.legend(fontsize=8.5, loc="upper right")

        modes = [
            ("с ограничениями", w_val, True, "o", "tab:blue"),
            ("без ограничений", w_val, False, "^", "tab:red"),
            ("wall=0, с ограничениями", 0.0, True, "s", "tab:green"),
            ("wall=0, без ограничений", 0.0, False, "v", "tab:purple"),
        ]

        for poly_item in building_polygons:
            poly = poly_item[0] if isinstance(poly_item, tuple) else poly_item
            if hasattr(poly, "exterior"):
                x, y = poly.exterior.xy
                ax_map.plot(x, y, color="gray", alpha=0.5, linewidth=1.0, zorder=1)

        all_x, all_y = [], []
        true_drawn = False

        for mode_label, w_cur, use_c, marker, color in modes:
            res = run_ap(ap_positions, building_edges, building_polygons, building_clusters, test_fp_data, r0, n_val, w_cur, use_c, 10)
            recs = [
                {"true": list(r["true_position"][:2]), "pred": list(r["predicted_position"][:2]), "err": r["error_meters"]}
                for r in res.values()
            ]
            m = calc_metrics([x["err"] for x in recs], total_test)
            csv_rows.append([
                set_name, mode_label,
                round(m["mean"], 2), round(m["median"], 2), round(m["rmse"], 2), round(m["p95"], 2),
                m["ok"], total_test, round(m["rate"], 1),
            ])
            print(
                f"{set_name:<18} | {mode_label:<28} | {m['mean']:<8.2f} | {m['median']:<10.2f} | "
                f"{m['rmse']:<8.2f} | {m['p95']:<8.2f} | {m['ok']}/{total_test} ({m['rate']:.0f}%)"
            )

            tx = [r["true"][0] for r in recs]
            ty = [r["true"][1] for r in recs]
            px = [r["pred"][0] for r in recs]
            py = [r["pred"][1] for r in recs]
            all_x.extend(tx + px)
            all_y.extend(ty + py)

            if not true_drawn and tx:
                ax_map.scatter(tx, ty, c="black", marker="*", s=190, label="Истинная позиция", zorder=5)
                for idx, (x0, y0) in enumerate(zip(tx, ty), start=1):
                    ax_map.annotate(str(idx), (x0, y0), textcoords="offset points", xytext=(5, 5), fontsize=8.5, fontweight="bold", zorder=6)
                true_drawn = True

            ax_map.scatter(px, py, c=color, marker=marker, s=80, alpha=0.85, label=mode_label, zorder=4)
            for x0, y0, x1, y1 in zip(tx, ty, px, py):
                ax_map.plot([x0, x1], [y0, y1], color=color, linestyle="--", linewidth=0.85, alpha=0.6)

            errs = np.sort([r["err"] for r in recs if r["err"] is not None and not np.isnan(r["err"])])
            if len(errs) > 0:
                cdf = np.arange(1, len(errs) + 1) / len(errs)
                ax_cdf.step(errs, cdf, where="post", color=color, linewidth=2, label=mode_label)

        if all_x and all_y:
            dx = max(max(all_x) - min(all_x), 1e-5)
            dy = max(max(all_y) - min(all_y), 1e-5)
            ax_map.set_xlim(min(all_x) - dx * 0.25, max(all_x) + dx * 0.25)
            ax_map.set_ylim(min(all_y) - dy * 0.25, max(all_y) + dy * 0.25)

        ax_map.xaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        ax_map.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        ax_map.tick_params(axis="x", rotation=20)
        ax_map.set_title(f"{set_name}: позиции на плане", fontsize=10)
        ax_map.set_xlabel("Долгота")
        ax_map.set_ylabel("Широта")
        ax_map.grid(True, linestyle=":", alpha=0.5)
        ax_map.legend(fontsize=8, loc="best")

        ax_cdf.set_title(f"{set_name}: распределение ошибок (CDF)", fontsize=10)
        ax_cdf.set_xlabel("Ошибка (м)")
        ax_cdf.set_ylabel("Доля точек")
        ax_cdf.grid(True, linestyle="--", alpha=0.6)
        ax_cdf.legend(fontsize=8.5, loc="lower right")

    csv_path = os.path.join(out_dir, "ap_train_vs_default.csv")
    png_path = os.path.join(out_dir, "ap_train_vs_default.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["param_set", "mode", "mean_m", "median_m", "rmse_m", "p95_m", "resolved", "total", "success_rate_pct"])
        writer.writerows(csv_rows)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print(f"\nСохранено в {out_dir}/: ap_train_vs_default.png, ap_train_vs_default.csv")
    plt.show()


if __name__ == "__main__":
    main()
