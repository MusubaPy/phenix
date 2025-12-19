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

Примечание: **`vanila`** — это baseline (используйте `build/bin/mjpc_vanila`), **`mod`** — наша модификация (используйте `build/bin/mjpc_mod`).

Примеры (быстрый quickstart):
```bash
# мод, прогон с CSV и 60s
MJPC_CSV_LOG=logs/one_off_run/run.csv MJPC_MAX_SIM_TIME=60 build/bin/mjpc_mod

# vanila baseline
MJPC_CSV_LOG=logs/vanila_run/run.csv MJPC_MAX_SIM_TIME=60 build/bin/mjpc_vanila
```

## Data pipeline — что и где 📁
- `scripts/convert_mjpc_csv.py` — делает `_conv.csv` с детерминированным LF-only header.
- `scripts/compute_metrics.py` — считает mean(pitch/roll) и mean(energy); по умолчанию требует `MJPC_MIN_TRAVEL_DISTANCE_M=5.0` (можно переопределить через env).
- `scripts/run_grf_variants.sh` — запускает варианты (A_perfoot, B_norm_mix, C_transition), кладёт outputs в `logs/sweep_grf/variants`.
- `scripts/collect_datasets.py` — универсальный сборщик (run→convert→metrics) и пишет `results.csv`.
- `scripts/run_full_comparison.py` — runner для baseline vs modified (по-умолчанию тримит [10s,60s]).

## Анализ результатов
- Быстрая оценка одного CSV (рекомендуется — использует интегрированную энергию):
```bash
python3 scripts/gait_evaluator.py --csv logs/<run>/run.csv --log logs/<run>/run.log --out logs/<run>/run_eval_fixed.json --plots logs/<run>/plots_fixed
```
- Важное замечание: используйте `energy_per_m_corrected` (интеграл torque_applied*joint_vel) для сравнения энергозатрат — оно более надёжное, чем сырое поле `energy_abs_j` в CSV.

## Важные env-переменные для экспериментов (актуальные) 🧪
    env.update({
        # Core experiment controls
        'MJPC_CSV_LOG': csv_path,
        'MJPC_MAX_SIM_TIME': '60',           # seconds per run
        # Fix seed to 1 unless explicitly changed via --seed or --extra-env.
        # Do NOT inherit MJPC_SEED from the caller environment to avoid
        # accidental variation during quick checks.
        'MJPC_SEED': str(seed if seed is not None else 1),

        # Cost/penalty knobs
        'MJPC_GRF_WEIGHT': '1e-5',              # global GRF cost scalar (0 disables)
        'MJPC_GRF_PER_FOOT_SCALE': '0.8,0.8,1.2,1.2',  # per-foot GRF scaling
        'MJPC_GRF_HIND_WEIGHT': '1e-7',
        'MJPC_GRF_FRONT_WEIGHT': '1e-7',
        'MJPC_INTERNAL_GRF_ALIGN_WEIGHT': env.get('MJPC_INTERNAL_GRF_ALIGN_WEIGHT', '1e-3'),
        'MJPC_TARGET_SMOOTH_TAU': str(tau),  # smoothing tau
        'MJPC_TARGET_BLEND_ALPHA': str(alpha),
        'MJPC_FIXATION_WEIGHT': env.get('MJPC_FIXATION_WEIGHT', '1e-4'),
        'MJPC_POWER_PENALTY_WEIGHT': env.get('MJPC_POWER_PENALTY_WEIGHT', '0'),
        'MJPC_POWER_PENALTY_MODE': env.get('MJPC_POWER_PENALTY_MODE', ''),
        'MJPC_HEIGHT_WEIGHT_SCALE': env.get('MJPC_HEIGHT_WEIGHT_SCALE', '1.0'),
        'MJPC_BIARTICULAR_GAIN': env.get('MJPC_BIARTICULAR_GAIN', '0.0'),

        # GRF control/misc
        'MJPC_GRF_TRANSITION_BOOST': env.get('MJPC_GRF_TRANSITION_BOOST', '0.0'),
        'MJPC_GRF_NORMALIZE': env.get('MJPC_GRF_NORMALIZE', '0'),
        'MJPC_GRF_LOSS_MIX': env.get('MJPC_GRF_LOSS_MIX', '0.0'),
        'MJPC_GRF_MOTOR_BLEND': env.get('MJPC_GRF_MOTOR_BLEND', '0.0'),
        'MJPC_DISABLE_MOTOR_BLEND': env.get('MJPC_DISABLE_MOTOR_BLEND', '0'),
        'MJPC_GRF_TARGET_MODE': env.get('MJPC_GRF_TARGET_MODE', ''),

        # Simulation / stability controls
        'MJPC_CONTACT_STABLE_STEPS': env.get('MJPC_CONTACT_STABLE_STEPS', '10'),
        'MJPC_CTRL_CLIP': env.get('MJPC_CTRL_CLIP', '100'),
        'MJPC_MIN_TRAVEL_DISTANCE_M': env.get('MJPC_MIN_TRAVEL_DISTANCE_M', '0'),

        # Sensor / debugging (advanced)
        'MJPC_GRF_SENSOR_SCALE': env.get('MJPC_GRF_SENSOR_SCALE', '1.0'),
        'MJPC_REFLECT_NET_FORCE_GAIN': env.get('MJPC_REFLECT_NET_FORCE_GAIN', '0.0'),

        # Metrics (post-processing helpers)
        'MJPC_METRICS_START_SEC': env.get('MJPC_METRICS_START_SEC', ''),
        'MJPC_METRICS_END_SEC': env.get('MJPC_METRICS_END_SEC', ''),
    })


## Ключевые места в кодовой базе
- `mjpc/` — основной C++ код (tasks, planners, residuals)
- `mjpc/tasks/quadruped_mod/` — место последних GRF-изменений (ResetLocked, Residual caching, env parsing)
- `build/bin/` — `mjpc`, `mjpc_mod`
- `scripts/` — pipeline: convert, metrics, sweeps, tests
- `logs/` — все результаты прогонов и summary CSV

---

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
