(Пока в README непроверенный нейрослоп, потом проверю)

# Локализация робота по сигналам Wi-Fi (RSSI, FTM/RTT, AoA)

Репозиторий содержит код, данные и результаты экспериментов по сравнению трёх подходов к локализации мобильного робота.

## Структура репозитория и описание файлов

### `task1_osmag/` — Локализация по уровню сигнала RSSI (`osmag-wifi-localization`)
Базовый репозиторий: https://github.com/XuMa369/osmag-wifi-localization/tree/main

* `task1_osmag/ap_localization.py` — калибровка модели затухания и оценка координат 48 точек доступа (`map/AP_MAP.osm`) исключительно по обучающей выборке (`map/wifi_data.osm`).
* `task1_osmag/KNN_localisation.py` — перебор гиперпараметров k-NN (5 метрик расстояния, 2 схемы взвешивания, k от 1 до 100) на тестовых точках `map/Non-FingerprintedAreas.osm`, расчёт метрик с пространственными ограничениями и без них, построение графиков и карты.
* `task1_osmag/AP_new_localisation.py` — сравнение физической AP-модели затухания на калиброванных (`train`) и стандартных (`default`) параметрах, исследование влияния учёта стен (`wall = train` vs `wall = 0`), пространственных ограничений и числа итераций (`iter = 1..15`).
* `task1_osmag/knn_all_metrics.csv` / `task1_osmag/knn_all_metrics.png` — таблица всех 1000 прогонов k-NN и гистограммы ошибок по метрикам.
* `task1_osmag/knn_map_positions.png` — карта здания с обучающими точками, истинными позициями теста и предсказаниями лучших конфигураций k-NN.
* `task1_osmag/ap_train_vs_default.csv` / `task1_osmag/ap_train_vs_default.png` — сводные метрики AP-модели, гистограммы по итерациям, карты позиций и графики CDF.

**Команды запуска (Задача 1):**

    cd task1_osmag
    python KNN_localisation.py
    python AP_new_localisation.py

---

### `task2_hymn/` — Локализация по дальностям Wi-Fi FTM/RTT (`HYMN-dataset`)
Базовый репозиторий: `TUD-ITVS/HYMN-dataset`.
* *(файлы и команды запуска будут добавлены после реализации WLS-локализатора)*

---

### `task3_wiros/` — Локализация по углам прихода сигнала AoA (`WiROS`)
Базовый репозиторий: `ucsdwcsng/WiROS`.
* *(файлы и команды запуска будут добавлены после проверки `forward_localization`)*
