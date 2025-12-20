#!/usr/bin/env bash
set -euo pipefail

# Sweep Alexander parameters (alex_power, alex_align, alex_fx_smooth)
# Step 0.1 for power and align in [0,1]. fx_smooth allowed values: 0 or 1e-9 (max stable)
# 3 repeats per combination. Each run: timeout 30s (headless), CSV log per run.

OUTDIR="logs/sweep_alex_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

# geometric grid: 1e-1 .. 1e-10
POWER_EXP_MIN=1
POWER_EXP_MAX=10
ALIGN_EXP_MIN=1
ALIGN_EXP_MAX=10

# fx values limited to 1e-9 and 1e-10 per instruction
FX_VALUES=(1e-9 1e-10)
REPEATS=2
SIMTIME=15

MJPC_BIN="$(pwd)/build/bin/mjpc_mod"
if [[ ! -x "$MJPC_BIN" ]]; then
  MJPC_BIN="build/bin/mjpc_mod"
fi

echo "Sweep output dir: $OUTDIR"
echo "Using mjpc binary: $MJPC_BIN"

# Detect whether xvfb-run is available; allow forcing no-xvfb with NO_XVFB=1
USE_XVFB=1
if [[ "${NO_XVFB:-0}" == "1" ]]; then
  USE_XVFB=0
fi
if ! command -v xvfb-run >/dev/null 2>&1; then
  USE_XVFB=0
  echo "[WARN] 'xvfb-run' not found in PATH — will fall back to running without Xvfb."
  echo "Set NO_XVFB=1 to silence this warning. To enable headless X, install 'xvfb' and 'xvfb-run' (e.g. apt install xvfb)."
fi

# Quick check mode: verify generated filenames without running simulations
if [[ "${1:-}" == "--check-names" ]]; then
  echo "Checking sweep filename generation (no simulation runs)..."
  declare -A seen
  dup_count=0
  bad_count=0
  names=()
  for pe in $(seq $POWER_EXP_MIN $POWER_EXP_MAX); do
    p=$(printf "1e-%d" "$pe")
    for ae in $(seq $ALIGN_EXP_MIN $ALIGN_EXP_MAX); do
      a=$(printf "1e-%d" "$ae")
      for fx in "${FX_VALUES[@]}"; do
        for r in $(seq 1 $REPEATS); do
          name=$(printf "p%s_a%s_fx%s_r%d.csv" "$p" "$a" "$fx" "$r")
          names+=("$name")
          if [[ -n "${seen[$name]:-}" ]]; then
            dup_count=$((dup_count+1))
            if [[ $dup_count -le 20 ]]; then
              echo "DUPLICATE: $name"
            fi
          fi
          seen[$name]=1
          # simple invalid char check (slash or space)
          if [[ "$name" == *"/"* || "$name" == *" "* ]]; then
            bad_count=$((bad_count+1))
            echo "INVALID NAME: $name"
          fi
        done
      done
    done
  done
  total=${#names[@]}
  unique=${#seen[@]}
  echo "Total names generated: $total"
  echo "Unique names: $unique"
  echo "Duplicates found: $((total - unique))"
  echo "Invalid names found: $bad_count"
  echo "Example names (first 12):"
  for i in $(seq 0 11); do
    echo "  ${names[$i]}"
  done
  exit 0
fi

for pe in $(seq $POWER_EXP_MIN $POWER_EXP_MAX); do
  p=$(printf "1e-%d" "$pe")
  for ae in $(seq $ALIGN_EXP_MIN $ALIGN_EXP_MAX); do
    a=$(printf "1e-%d" "$ae")
    for fx in "${FX_VALUES[@]}"; do
      for r in $(seq 1 $REPEATS); do
        # keep the '1e-#' formatting intact by using string-format
        name=$(printf "p%s_a%s_fx%s_r%d" "$p" "$a" "$fx" "$r")
        csv="$OUTDIR/${name}.csv"
        echo "Running: power=$p align=$a fx=$fx repeat=$r -> $csv"
        if [[ "$USE_XVFB" == "1" ]]; then
          xvfb-run -a -s "-screen 0 1280x1024x24" env MJPC_CSV_LOG="$csv" timeout ${SIMTIME}s "$MJPC_BIN" \
            --task "Quadruped Flat (mod)" \
            --alex_enabled \
            --alex_power_weight=${p} \
            --alex_align_weight=${a} \
            --alex_fx_smooth_weight=${fx} || true
        else
          echo "[WARN] Running without Xvfb (may open a window or fail on headless machine)"
          env MJPC_CSV_LOG="$csv" timeout ${SIMTIME}s "$MJPC_BIN" \
            --task "Quadruped Flat (mod)" \
            --alex_enabled \
            --alex_power_weight=${p} \
            --alex_align_weight=${a} \
            --alex_fx_smooth_weight=${fx} || true
        fi

        # verify CSV was produced and is not empty; log failures for rerun
        if [[ ! -s "$csv" ]]; then
          echo "$(date +%s) $p $a $fx $r" >> "$OUTDIR/failed_runs.txt"
          echo "[WARN] Run did not produce CSV or file empty: $csv"
        else
          echo "$csv" >> "$OUTDIR/completed_runs.txt"
        fi
      done
    done
  done
done

echo "Sweep runs finished. CSVs in: $OUTDIR"
      echo "Computing raw per-run metrics (this may take a moment)..."
      python3 scripts/compute_mech_heat.py --csv $OUTDIR/*.csv --tmin 5 --out $OUTDIR/mech_heat_raw.json
      echo "Aggregating runs by parameter combination..."
      python3 scripts/aggregate_sweep.py --raw $OUTDIR/mech_heat_raw.json --out $OUTDIR/summary_agg.json
      echo "Top combos stored in $OUTDIR/summary_agg.json"

      echo "Also running 3 baseline runs for mjpc_vanila (for statistics)..."
      for r in 1 2 3; do
        vcsv="$OUTDIR/vanilla_r${r}.csv"
        echo "Vanilla run $r -> $vcsv"
        if [[ "$USE_XVFB" == "1" ]]; then
          xvfb-run -a -s "-screen 0 1280x1024x24" env MJPC_CSV_LOG="$vcsv" timeout ${SIMTIME}s "$MJPC_BIN" --task "Quadruped Flat" || true
        else
          echo "[WARN] Running vanilla without Xvfb"
          env MJPC_CSV_LOG="$vcsv" timeout ${SIMTIME}s "$MJPC_BIN" --task "Quadruped Flat" || true
        fi
        if [[ ! -s "$vcsv" ]]; then
          echo "$(date +%s) vanilla $r" >> "$OUTDIR/failed_runs.txt"
          echo "[WARN] Vanilla run did not produce CSV: $vcsv"
        else
          echo "$vcsv" >> "$OUTDIR/completed_runs.txt"
        fi
      done
      echo "Computing vanilla metrics..."
      python3 scripts/compute_mech_heat.py --csv $OUTDIR/vanilla_r*.csv --tmin 5 --out $OUTDIR/mech_heat_vanilla_raw.json || true

      echo "Aggregating vanilla with others..."
      python3 scripts/aggregate_sweep.py --raw $OUTDIR/mech_heat_vanilla_raw.json --out $OUTDIR/summary_vanilla.json || true

      echo "Done. Summary files: $OUTDIR/mech_heat_raw.json, $OUTDIR/summary_agg.json, $OUTDIR/summary_vanilla.json"
