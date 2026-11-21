# Локализация робота по сигналам Wi-Fi (RSSI, FTM/RTT, AoA)

Научно-проектная работа по сравнению трёх подходов к локализации мобильного робота в помещениях по сигналам Wi-Fi.

## 1. `task1_osmag/` — Локализация по RSSI
Исходный репозиторий: [https://github.com/XuMa369/osmag-wifi-localization](https://github.com/XuMa369/osmag-wifi-localization)

* `ap_localization.py` — оценка координат точек доступа на обучающей выборке.
* `AP_new_localisation.py` — сравнение модели затухания со стенами и без них.
* `KNN_localisation.py` — перебор параметров k-NN с пространственными границами и без.
* `robot_AP_localization.py` — базовый запуск локализации по модели точек доступа.
* `robot_fingerprint_localization.py` — базовый запуск k-NN локализации по радиоотпечаткам.
* `AP_localisation/` — графики и таблица метрик модели по точкам доступа.
* `KNN_localisation/` — графики, карта и таблицы метрик метода k-NN.

## 2. `task2_hymn/` — Локализация по FTM/RTT
Исходный репозиторий: [https://github.com/TUD-ITVS/HYMN-dataset](https://github.com/TUD-ITVS/HYMN-dataset)

* `inspect_format.py` — проверка формата данных, высот и порядка столбцов дальностей.
* `OLS_localisation.py` — 2D и 3D МНК локализация с калибровкой смещения.
* `OLS_delete_one_ap_localisation.py` — проверка точности 2D МНК при отключении одной AP.
* `OLS_localisation/` — карта стоянок, кривые CDF и метрики 2D/3D МНК.
* `OLS_leave_one_ap_out/` — таблицы и графики CDF при отключении точек доступа.

## 3. `task3_wiros/` — Локализация по AoA
Исходный репозиторий: [https://github.com/ucsdwcsng/WiROS](https://github.com/ucsdwcsng/WiROS)

* `WIROS/` — исходные ROS-пакеты WiROS и датасет `forward_localization`.
* `inspect_wiros.py` — восстановление эталонной траектории и статистика ошибок углов.
* `OLS_localisation.py` — локализация линейным МНК по пересечению лучей AoA.
* `OLS_nonlinear_localisation.py` — локализация нелинейным МНК по угловым невязкам.
* `OLS_delete_one_ap_localisation.py` — сравнение линейного и нелинейного МНК без одной AP.
* `synthetic_noise_experiment.py` — тест устойчивости алгоритмов к синтетическому угловому шуму.
* `wiros_inspect/` — эталонная траектория, графики и статистика угловых ошибок.
* `OLS_localisation/` — метрики, карта и CDF линейного МНК.
* `OLS_nonlinear_localisation/` — метрики, карта и CDF нелинейного МНК.
* `OLS_leave_one_ap_out/` — сравнительные таблицы и графики при отключении AP.
* `OLS_synthetic_noise/` — графики и метрики эксперимента с синтетическим шумом.
