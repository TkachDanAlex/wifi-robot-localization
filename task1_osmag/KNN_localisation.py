import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.neighbors import KNeighborsRegressor

from io_layer.osm_parser import OsmDataParser, WiFiFingerprintParser, WiFiTestPointParser
from core.fingerprint_knn import WiFiKNNLocalizer


def calc_metrics(errors, total_count):
    valid = [float(e) for e in errors if e is not None and not np.isnan(e)]
    if not valid:
        return {"mean": np.nan, "median": np.nan, "rmse": np.nan, "p95": np.nan, "ok": 0, "rate": 0.0}
    arr = np.array(valid, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "p95": float(np.percentile(arr, 95)),
        "ok": len(arr),
        "rate": float(len(arr) / total_count * 100.0),
    }


def extract_unconstrained_error(rec):
    if "error_unconstrained" in rec and rec["error_unconstrained"] is not None:
        return rec["error_unconstrained"]
    return rec.get("error_constrained")


def main():
    out_dir = "KNN_localisation"
    os.makedirs(out_dir, exist_ok=True)

    fp_parser = WiFiFingerprintParser("./map/wifi_data.osm")
    fingerprint_data = fp_parser.parse_fingerprint_data()

    tp_parser = WiFiTestPointParser("./map/Non-FingerprintedAreas.osm")
    test_points = tp_parser.parse_test_points()
    total_test = len(test_points)

    localizer_const = WiFiKNNLocalizer(
        fingerprint_data=fingerprint_data,
        all_macs=fp_parser.all_macs,
        k=5,
        polygon_file="./map/wifi_data.osm",
        use_boundary_constraint=True,
    )

    localizer_unconst = WiFiKNNLocalizer(
        fingerprint_data=fingerprint_data,
        all_macs=fp_parser.all_macs,
        k=5,
        polygon_file=None,
        use_boundary_constraint=False,
    )

    max_k = min(100, len(localizer_const.feature_matrix))
    print(f"Точек в базе: {len(localizer_const.feature_matrix)}, тестовых точек: {total_test}, диапазон k: 1..{max_k}")

    metrics_list = [
        ("euclidean", "L2 (евклидова)"),
        ("manhattan", "L1 (манхэттенская)"),
        ("cosine", "Косинусная"),
        ("chebyshev", "L-inf (Чебышёва)"),
        ("correlation", "Корреляционная"),
    ]
    weights_list = [
        ("uniform", "вес: 1"),
        ("distance", "вес: 1/d"),
    ]

    variants = {}
    for m_key, m_label in metrics_list:
        for w_key, w_label in weights_list:
            variants[(m_key, w_key)] = {
                "title": f"{m_label} | {w_label}",
                "ks": [],
                "mean": [],
                "median": [],
                "rmse": [],
                "p95": [],
                "best_res_const": None,
                "best_res_unconst": None,
                "best_k": None,
                "best_mean": np.inf,
            }

    csv_rows = []

    for m_key, _ in metrics_list:
        for w_key, _ in weights_list:
            v = variants[(m_key, w_key)]
            for k in range(1, max_k + 1):
                localizer_const.k = k
                localizer_const.knn_model = KNeighborsRegressor(n_neighbors=k, weights=w_key, metric=m_key)
                localizer_const.knn_model.fit(localizer_const.feature_matrix, localizer_const.position_matrix)

                res_c = localizer_const.localize(test_points)
                errs_c = [r.get("error_constrained") for r in res_c.values()]
                m_c = calc_metrics(errs_c, total_test)

                v["ks"].append(k)
                v["mean"].append(m_c["mean"])
                v["median"].append(m_c["median"])
                v["rmse"].append(m_c["rmse"])
                v["p95"].append(m_c["p95"])

                if m_c["mean"] < v["best_mean"]:
                    v["best_mean"] = m_c["mean"]
                    v["best_k"] = k
                    v["best_res_const"] = res_c

                csv_rows.append([
                    m_key, w_key, "with_constraints", k,
                    round(m_c["mean"], 2), round(m_c["median"], 2), round(m_c["rmse"], 2), round(m_c["p95"], 2),
                    m_c["ok"], total_test, round(m_c["rate"], 1),
                ])

    with open(os.path.join(out_dir, "knn_all_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "weights", "mode", "k", "mean_m", "median_m", "rmse_m", "p95_m", "resolved", "total", "success_rate_pct"])
        writer.writerows(csv_rows)

    comp_rows = []
    print(f"\n{'Вариант':<30} | {'Best k':<6} | {'Режим':<18} | {'Mean (м)':<8} | {'Median (м)':<10} | {'RMSE (м)':<8} | {'95% (м)':<8} | {'Решений':<12}")
    print("-" * 118)

    for (m_key, w_key), v in variants.items():
        best_k = v["best_k"]

        localizer_unconst.k = best_k
        localizer_unconst.knn_model = KNeighborsRegressor(n_neighbors=best_k, weights=w_key, metric=m_key)
        localizer_unconst.knn_model.fit(localizer_unconst.feature_matrix, localizer_unconst.position_matrix)
        res_u = localizer_unconst.localize(test_points)
        v["best_res_unconst"] = res_u

        errs_c = [r.get("error_constrained") for r in v["best_res_const"].values()]
        errs_u = [extract_unconstrained_error(r) for r in res_u.values()]

        m_c = calc_metrics(errs_c, total_test)
        m_u = calc_metrics(errs_u, total_test)

        for mode_name, m_obj in [("с ограничениями", m_c), ("без ограничений", m_u)]:
            print(
                f"{v['title']:<30} | {best_k:<6} | {mode_name:<18} | {m_obj['mean']:<8.2f} | "
                f"{m_obj['median']:<10.2f} | {m_obj['rmse']:<8.2f} | {m_obj['p95']:<8.2f} | "
                f"{m_obj['ok']}/{total_test} ({m_obj['rate']:.0f}%)"
            )
            comp_rows.append([
                v["title"], best_k, mode_name,
                round(m_obj["mean"], 2), round(m_obj["median"], 2), round(m_obj["rmse"], 2), round(m_obj["p95"], 2),
                m_obj["ok"], total_test, round(m_obj["rate"], 1),
            ])

    with open(os.path.join(out_dir, "knn_constraints_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "best_k", "mode", "mean_m", "median_m", "rmse_m", "p95_m", "resolved", "total", "success_rate_pct"])
        writer.writerows(comp_rows)

    fig, axes = plt.subplots(5, 2, figsize=(15, 19), sharey=True)
    axes = axes.flatten()

    for ax, v in zip(axes, variants.values()):
        ks = v["ks"]
        best_idx = int(np.nanargmin(v["mean"]))
        best_k = ks[best_idx]
        best_mean = v["mean"][best_idx]

        bars = ax.bar(ks, v["mean"], color="steelblue", alpha=0.75, width=0.8, label="Mean (м)")
        bars[best_idx].set_color("crimson")

        ax.plot(ks, v["rmse"], color="darkorange", linewidth=1.5, linestyle="--", label="RMSE (м)")
        ax.plot(ks, v["p95"], color="purple", linewidth=1.5, linestyle="-.", label="95% (м)")

        ax.set_title(f"{v['title']} (мин. Mean = {best_mean:.2f} м при k={best_k})", fontsize=9.5)
        ax.set_xlabel("k", fontsize=8.5)
        ax.set_ylabel("Ошибка (м)", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "knn_all_metrics.png"), dpi=200)

    fig_cdf, ax_cdf = plt.subplots(figsize=(9.5, 5.5))
    cdf_specs = [
        (("correlation", "distance"), "best_res_const", "error_constrained", "tab:blue", "-", "Корреляционная (1/d, с границами)"),
        (("correlation", "distance"), "best_res_unconst", "unconstrained", "tab:red", "--", "Корреляционная (1/d, без границ)"),
        (("euclidean", "distance"), "best_res_const", "error_constrained", "tab:orange", "-", "L2 евклидова (1/d, с границами)"),
        (("euclidean", "distance"), "best_res_unconst", "unconstrained", "tab:purple", "--", "L2 евклидова (1/d, без границ)"),
    ]
    for key, res_field, err_mode, color, ls, label_prefix in cdf_specs:
        v = variants[key]
        res = v[res_field]
        if err_mode == "unconstrained":
            errs = np.sort([extract_unconstrained_error(r) for r in res.values() if extract_unconstrained_error(r) is not None])
        else:
            errs = np.sort([r[err_mode] for r in res.values() if r.get(err_mode) is not None])
        if len(errs) > 0:
            cdf = np.arange(1, len(errs) + 1) / len(errs)
            ax_cdf.step(errs, cdf, where="post", color=color, linestyle=ls, linewidth=2.0, label=f"{label_prefix}, k={v['best_k']}")

    ax_cdf.axhline(0.95, color="gray", linestyle=":", linewidth=1.1, label="Порог 95%")
    ax_cdf.set_title("k-NN: сравнение с пространственными ограничениями и без них (CDF)")
    ax_cdf.set_xlabel("Ошибка (м)")
    ax_cdf.set_ylabel("Доля тестовых точек")
    ax_cdf.grid(True, linestyle="--", alpha=0.6)
    ax_cdf.legend(loc="lower right", fontsize=8.5)
    fig_cdf.tight_layout()
    fig_cdf.savefig(os.path.join(out_dir, "knn_constraints_cdf.png"), dpi=200)

    map_file = "./map/base_map.osm" if os.path.exists("./map/base_map.osm") else "./map/wifi_data.osm"
    poly_parser = OsmDataParser(map_file)
    poly_parser.parse()
    poly_parser.extract_polygon_edges()
    _, _, _, _, _, building_polygons, _, _ = poly_parser.get_data()

    fig_map, ax_map = plt.subplots(figsize=(10, 8))

    for poly_item in building_polygons:
        poly = poly_item[0] if isinstance(poly_item, tuple) else poly_item
        if hasattr(poly, "exterior"):
            x, y = poly.exterior.xy
            ax_map.plot(x, y, color="gray", alpha=0.5, linewidth=1.0, zorder=1)

    train_pos = localizer_const.position_matrix
    ax_map.scatter(
        train_pos[:, 0],
        train_pos[:, 1],
        c="deepskyblue",
        s=14,
        alpha=0.35,
        label="Обучающая база (wifi_data.osm)",
        zorder=2,
    )

    map_variants = [
        (("euclidean", "distance"), "predicted_position_constrained", "o", "tab:orange", "L2 (1/d, с границами)"),
        (("cosine", "distance"), "predicted_position_constrained", "s", "tab:green", "Косинусная (1/d, с границами)"),
        (("correlation", "distance"), "predicted_position_constrained", "^", "tab:blue", "Корреляционная (1/d, с границами)"),
        (("correlation", "distance"), "predicted_position_unconstrained", "v", "tab:red", "Корреляционная (1/d, без границ)"),
    ]

    all_x, all_y = [], []
    true_drawn = False

    for key, pos_field, marker, color, label_prefix in map_variants:
        v = variants[key]
        res = v["best_res_const"]
        if not res:
            continue

        tx = [r["true_position"][0] for r in res.values()]
        ty = [r["true_position"][1] for r in res.values()]
        px = [r[pos_field][0] for r in res.values()]
        py = [r[pos_field][1] for r in res.values()]
        all_x.extend(tx + px)
        all_y.extend(ty + py)

        if not true_drawn and tx:
            ax_map.scatter(tx, ty, c="black", marker="*", s=220, label="Истинная позиция (тест)", zorder=6)
            for idx, (x0, y0) in enumerate(zip(tx, ty), start=1):
                ax_map.annotate(
                    str(idx),
                    (x0, y0),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=9.5,
                    fontweight="bold",
                    zorder=7,
                )
            true_drawn = True

        ax_map.scatter(
            px,
            py,
            c=color,
            marker=marker,
            s=85,
            alpha=0.85,
            label=f"{label_prefix}, k={v['best_k']}",
            zorder=5,
        )
        for x0, y0, x1, y1 in zip(tx, ty, px, py):
            ax_map.plot([x0, x1], [y0, y1], color=color, linestyle="--", linewidth=0.9, alpha=0.6, zorder=4)

    if all_x and all_y:
        dx = max(max(all_x) - min(all_x), 1e-5)
        dy = max(max(all_y) - min(all_y), 1e-5)
        ax_map.set_xlim(min(all_x) - dx * 0.35, max(all_x) + dx * 0.35)
        ax_map.set_ylim(min(all_y) - dy * 0.35, max(all_y) + dy * 0.35)

    ax_map.xaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax_map.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax_map.tick_params(axis="x", rotation=20)
    ax_map.set_title("Позиции k-NN на плане здания (область теста)")
    ax_map.set_xlabel("Долгота")
    ax_map.set_ylabel("Широта")
    ax_map.grid(True, linestyle=":", alpha=0.5)
    ax_map.legend(fontsize=8.5, loc="best")

    fig_map.tight_layout()
    fig_map.savefig(os.path.join(out_dir, "knn_map_positions.png"), dpi=200)

    print(f"\nСохранено в {out_dir}/: knn_all_metrics.png, knn_constraints_cdf.png, knn_map_positions.png, knn_all_metrics.csv, knn_constraints_comparison.csv")
    plt.show()


if __name__ == "__main__":
    main()
