# Инструкция для Copilot (актуально)

Пиши по-русски; делай минимальные и корректные правки C++/Python/shell, сохраняй сборку и тесты в рабочем состоянии. Избегай крупных архитектурных изменений без обсуждения.

## Быстрая сборка ✅
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build . 
```

## Быстрые запуски — что важно знать 🔧
- Бинар основного baseline задания (GUI): `build/bin/mjpc_vanila`
- Модифицированный бинарь (GRF-related changes): `build/bin/mjpc_mod` — скрипты предпочитают его, если он есть
- Пример одиночного прогона с логированием в CSV:
```bash
MJPC_CSV_LOG=logs/run.csv MJPC_MAX_SIM_TIME=45 build/bin/mjpc_mod --task="Quadruped Flat (mod)"
```
- Пример запуска GRF-sweep (варианты A/B/C):
```bash
BUILD_BIN=build/bin/mjpc_mod bash scripts/run_grf_variants.sh 1e-8 3
```

## Тесты и smoke checks ✅
- Запуск smoke tests:
```bash
python3 -m pytest scripts/tests/test_smoke_pipeline.py
```
- Короткие проверки, которые должны проходить:
  - `scripts/convert_mjpc_csv.py` — header строки LF-only
  - `scripts/compute_metrics.py` — выводит pitch, roll, energy; поддерживает `MJPC_MIN_TRAVEL_DISTANCE_M`

## Data pipeline — что и где 📁
- `scripts/convert_mjpc_csv.py` — делает `_conv.csv` с детерминированным LF-only header.
- `scripts/compute_metrics.py` — считает mean(pitch/roll) и mean(energy); по умолчанию требует `MJPC_MIN_TRAVEL_DISTANCE_M=5.0` (можно переопределить через env).
- `scripts/run_grf_variants.sh` — запускает варианты (A_perfoot, B_norm_mix, C_transition), кладёт outputs в `logs/sweep_grf/variants`.
- `scripts/collect_datasets.py` — универсальный сборщик (run→convert→metrics) и пишет `results.csv`.
- `scripts/run_full_comparison.py` — runner для baseline vs modified (по-умолчанию тримит [10s,60s]).

## Важные env-переменные для экспериментов (актуальные) 🧪
- `MJPC_GRF_WEIGHT`
- `MJPC_GRF_PER_FOOT_SCALE` (формат `f0,f1,f2,f3`)
- `MJPC_GRF_TRANSITION_BOOST`
- `MJPC_GRF_NORMALIZE` (0/1)
- `MJPC_GRF_LOSS_MIX` (0..1)
- `MJPC_INTERNAL_GRF_ALIGN_WEIGHT` (internal override)
- `MJPC_MIN_TRAVEL_DISTANCE_M` (по умолчанию 5.0, можно временно установить 0 для отладки)
- `MJPC_MAX_SIM_TIME` (s)
- `MJPC_CSV_LOG` (куда писать лог)

## Логи и результаты — где искать
- Sweep outputs: `logs/sweep_grf/variants` (raw CSV + `_conv.csv`)
- Baseline vs modified: `logs/baseline_vs_mod/results.csv`
- Если `compute_metrics.py` возвращает `NO_TRAVEL`, обычно причина — симуляция не прошла достаточное расстояние; сначала попробуйте увеличить `MJPC_MAX_SIM_TIME`, либо временно `MJPC_MIN_TRAVEL_DISTANCE_M=0` для быстрой проверки.

## Ключевые места в кодовой базе
- `mjpc/` — основной C++ код (tasks, planners, residuals)
- `mjpc/tasks/quadruped_mod/` — место последних GRF-изменений (ResetLocked, Residual caching, env parsing)
- `build/bin/` — `mjpc`, `mjpc_mod`
- `scripts/` — pipeline: convert, metrics, sweeps, tests
- `logs/` — все результаты прогонов и summary CSV

---
Если нужно, добавлю шаблоны команд для воспроизведения полного sweep с рекомендуемыми env и N≥3, а также чеклист для отлова нестабильных прогонов (NaN / Rollout divergence).
# Инструкция для Copilot

Говори по-русски. Пиши и правь C++-код в рамках текущей архитектуры, делай его понятным и лаконичным. Поддерживай компиляцию и запуск без ошибок, добавляй недостающие зависимости и согласовывай сопряженные файлы. Минимизируй изменения архитектуры.

## Рабочий цикл
- Планируй решения на основе структуры репозитория.
- Вноси правки итеративно до полной готовности задачи.
- После изменений собирай и проверяй проект:

```
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build . --config=Release
```

## Рабочая папка
```
/phenix_ws/mujoco_mpc/mjpc/tasks/quadruped
```

## Контекст задачи
Реализуется новая cost-функция для шагающего робота A1 в плоском мире `quadruped/task_flat_(*).xml` (mod или vanila), цель — минимизировать энергозатраты на передвижение:
- Минимизировать ошибку позиционирования: состояние робота на шаге i+1 максимально соответствует целевому.
- Контролировать векторы сил реакции опоры (GRF) на каждой лапе: суммарный GRF поддерживает вес тела.
- Для задних лап в контакте минимизировать угол между вектором GRF и вектором из точки контакта к ближайшему мотору (по нормали).
- Для передних лап в контакте обеспечивать, чтобы суммарная сила реакции опоры (net force) была [0 0 mg].
