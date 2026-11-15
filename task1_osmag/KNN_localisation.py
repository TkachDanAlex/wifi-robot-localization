import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.neighbors import KNeighborsRegressor

from io_layer.osm_parser import OsmDataParser, WiFiFingerprintParser, WiFiTestPointParser
from core.fingerprint_knn import WiFiKNNLocalizer


def calc_metrics(errors):
    arr = np.array(errors, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "p95": float(np.percentile(arr, 95)),
    }


def main():
    fp_parser = WiFiFingerprintParser("./map/wifi_data.osm")
    fingerprint_data = fp_parser.parse_fingerprint_data()

    tp_parser = WiFiTestPointParser("./map/Non-FingerprintedAreas.osm")
    test_points = tp_parser.parse_test_points()

    localizer = WiFiKNNLocalizer(
        fingerprint_data=fingerprint_data,
        all_macs=fp_parser.all_macs,
        k=5,
        polygon_file="./map/wifi_data.osm",
        use_boundary_constraint=True,
    )

    max_k = min(100, len(localizer.feature_matrix))
    print(f"Точек в базе: {len(localizer.feature_matrix)}, диапазон k: 1..{max_k}")

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
                "best_res": None,
                "best_k": None,
                "best_mean": np.inf,
            }

    csv_rows = []

    for m_key, _ in metrics_list:
        for w_key, _ in weights_list:
            v = variants[(m_key, w_key)]
            for k in range(1, max_k + 1):
                localizer.k = k
                localizer.knn_model = KNeighborsRegressor(
                    n_neighbors=k,
                    weights=w_key,
                    metric=m_key,
                )
                localizer.knn_model.fit(localizer.feature_matrix, localizer.position_matrix)

                results = localizer.localize(test_points)
                errs = [r["error_constrained"] for r in results.values()]
                m = calc_metrics(errs)

                v["ks"].append(k)
                v["mean"].append(m["mean"])
                v["median"].append(m["median"])
                v["rmse"].append(m["rmse"])
                v["p95"].append(m["p95"])

                if m["mean"] < v["best_mean"]:
                    v["best_mean"] = m["mean"]
                    v["best_k"] = k
                    v["best_res"] = results

                csv_rows.append([m_key, w_key, k, m["mean"], m["median"], m["rmse"], m["p95"]])

    with open("knn_all_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "weights", "k", "mean_m", "median_m", "rmse_m", "p95_m"])
        writer.writerows(csv_rows)

    print(f"\n{'Вариант':<32} | {'Best k':<6} | {'Mean (м)':<8} | {'RMSE (м)':<8} | {'95% (м)':<8}")
    print("-" * 74)
    for v in variants.values():
        best_idx = int(np.nanargmin(v["mean"]))
        print(
            f"{v['title']:<32} | {v['ks'][best_idx]:<6} | "
            f"{v['mean'][best_idx]:<8.2f} | {v['rmse'][best_idx]:<8.2f} | {v['p95'][best_idx]:<8.2f}"
        )

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
    plt.savefig("knn_all_metrics.png", dpi=200)

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

    train_pos = localizer.position_matrix
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
        res = v["best_res"]
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
    fig_map.savefig("knn_map_positions.png", dpi=200)

    print("\nСохранено: knn_all_metrics.png, knn_map_positions.png, knn_all_metrics.csv")
    plt.show()


if __name__ == "__main__":
    main()
    