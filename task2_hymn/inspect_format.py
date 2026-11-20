import ast
import numpy as np
import pandas as pd
anchors = pd.read_csv("data/reference/csv/anchor_coordinates.csv")
wifi_ap = anchors[anchors["point_id"].str.startswith("WIFI_")].sort_values("point_id").reset_index(drop=True)
points = pd.read_csv("data/reference/csv/point_coordinates.csv")
wifi = pd.read_csv("data/processed/csv/wifi.csv")
print("--- Координаты и высоты 6 Wi-Fi роутеров (метры) ---")
print(wifi_ap[["point_id", "X_LOCAL", "Y_LOCAL", "Z_LOCAL"]].to_string(index=False))
p0 = points[points["point_id"] == "A13B6"].iloc[0]
print("\n--- Смещение смартфона на плите (WIFI - CENTER) ---")
print(f"dx = {p0['X_LOCAL_WIFI'] - p0['X_LOCAL_CENTER']:+.3f} м, "
      f"dy = {p0['Y_LOCAL_WIFI'] - p0['Y_LOCAL_CENTER']:+.3f} м, "
      f"dz = {p0['Z_LOCAL_WIFI'] - p0['Z_LOCAL_CENTER']:+.3f} м")
row0 = wifi.iloc[0]
measured_r = np.array(ast.literal_eval(row0["ranges"].replace("nan", "None")), dtype=float)
rx_xyz = np.array([row0["X_LOCAL_WIFI"], row0["Y_LOCAL_WIFI"], row0["Z_LOCAL_WIFI"]])
ap_xyz = wifi_ap[["X_LOCAL", "Y_LOCAL", "Z_LOCAL"]].values
true_3d = np.linalg.norm(ap_xyz - rx_xyz, axis=1)
print(f"\n--- Первая строка wifi.csv (точка {row0['point_id']}) ---")
print("Измеренные ranges в файле:", np.round(measured_r, 3))
print("Истинные 3D до WIFI_01..06:", np.round(true_3d, 3))
order = [4, 1, 0, 2, 3, 5]
print("После перестановки столбцов:", np.round(measured_r[order], 3))
print("Разница (измер - истина):    ", np.round(measured_r[order] - true_3d, 3))
