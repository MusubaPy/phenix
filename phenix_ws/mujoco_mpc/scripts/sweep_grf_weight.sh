#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BUILD_BIN:-$ROOT_DIR/bin/mjpc}"
LOG_DIR="$ROOT_DIR/logs/sweep_grf"
mkdir -p "$LOG_DIR"

# default sweep (adjust as desired)
WEIGHTS=(0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1.0)

if [[ $# -gt 0 ]]; then
    WEIGHTS=($@)
fi

RESULTS_CSV="$LOG_DIR/results.csv"
rm -f "$RESULTS_CSV"
# columns: weight,metrics, baseline_metrics, deltas, paths
echo "weight,margin_pitch_mean,margin_roll_mean,energy_abs_mean,baseline_margin_pitch_mean,baseline_margin_roll_mean,baseline_energy_abs_mean,delta_pitch,delta_roll,delta_energy,csv,baseline_csv" > "$RESULTS_CSV"

# Run a single baseline (no GRF cost) to compare against
BASE_CSV="$LOG_DIR/run_no_grf.csv"
BASE_CONV="$LOG_DIR/run_no_grf_conv.csv"
BASE_METRICS="NA"
if [[ ! -f "$BASE_CONV" ]]; then
    echo "Running baseline (no GRF cost) -> $BASE_CSV"
    MJPC_CSV_LOG="$BASE_CSV" MJPC_MAX_SIM_TIME=45 MJPC_GRF_WEIGHT=0 "$BIN" --task="Quadruped Flat"
    echo "Converting baseline $BASE_CSV -> $BASE_CONV"
    python3 "$ROOT_DIR/scripts/convert_mjpc_csv.py" "$BASE_CSV" "$BASE_CONV"
    BASE_METRICS=$(python3 "$ROOT_DIR/scripts/compute_metrics.py" "$BASE_CONV" 2>&1 || true)
else
    echo "Using existing baseline conversion $BASE_CONV"
    BASE_METRICS=$(python3 "$ROOT_DIR/scripts/compute_metrics.py" "$BASE_CONV" 2>&1 || true)
fi
if [[ "$BASE_METRICS" == "NO_DATA" ]]; then
    echo "Baseline: NO_DATA"
    BASE_METRICS="NA"
elif [[ "$BASE_METRICS" == NO_TRAVEL* ]]; then
    echo "Baseline: NO_TRAVEL"
    BASE_METRICS="NA"
else
    echo "Baseline metrics: $BASE_METRICS"
fi
if [[ "$BASE_METRICS" != "NA" ]]; then
    BASE_MP=$(echo "$BASE_METRICS" | cut -d',' -f1)
    BASE_MR=$(echo "$BASE_METRICS" | cut -d',' -f2)
    BASE_EN=$(echo "$BASE_METRICS" | cut -d',' -f3)
else
    BASE_MP="NA"
    BASE_MR="NA"
    BASE_EN="NA"
fi

for w in "${WEIGHTS[@]}"; do
    # sanitize weight for filename
    wfile=$(echo "$w" | sed 's/\./p/g; s/\-//g; s/1e/1e/g')
    CSV="$LOG_DIR/run_w${wfile}.csv"
    CONV="$LOG_DIR/run_w${wfile}_conv.csv"

    echo "Running weight=$w -> $CSV"
    MJPC_CSV_LOG="$CSV" MJPC_MAX_SIM_TIME=45 MJPC_GRF_WEIGHT="$w" "$BIN" --task="Quadruped Flat"
    if [[ ! -f "$CSV" ]]; then
        echo "ERROR: expected CSV $CSV not found" >&2
        exit 1
    fi

    echo "Converting $CSV -> $CONV"
    python3 "$ROOT_DIR/scripts/convert_mjpc_csv.py" "$CSV" "$CONV"

    echo -n "Computing metrics for weight=$w... "
    out=$(python3 "$ROOT_DIR/scripts/compute_metrics.py" "$CONV" || true)
    if [[ "$out" == "NO_DATA" ]]; then
        echo "NO DATA (likely insufficient runtime after warmup)"
        echo "$w,NA,NA,NA,${BASE_MP},${BASE_MR},${BASE_EN},NA,NA,NA,${CONV},${BASE_CONV}" >> "$RESULTS_CSV"
        continue
    fi
    if [[ "$out" == NO_TRAVEL* ]]; then
        echo "NO TRAVEL (robot didn't move enough)"
        echo "$w,NA,NA,NA,${BASE_MP},${BASE_MR},${BASE_EN},NA,NA,NA,${CONV},${BASE_CONV}" >> "$RESULTS_CSV"
        continue
    fi
    echo "$out"
    # out is three comma-separated fields
    mp=$(echo "$out" | cut -d',' -f1)
    mr=$(echo "$out" | cut -d',' -f2)
    en=$(echo "$out" | cut -d',' -f3)
    if [[ "$BASE_MP" != "NA" ]]; then
        dp=$(awk -v a="$mp" -v b="$BASE_MP" 'BEGIN{printf "%.6f", a-b}')
        dr=$(awk -v a="$mr" -v b="$BASE_MR" 'BEGIN{printf "%.6f", a-b}')
        de=$(awk -v a="$en" -v b="$BASE_EN" 'BEGIN{printf "%.6f", a-b}')
    else
        dp="NA"; dr="NA"; de="NA"
    fi
    echo "$w,$mp,$mr,$en,${BASE_MP},${BASE_MR},${BASE_EN},$dp,$dr,$de,$CONV,${BASE_CONV}" >> "$RESULTS_CSV"

    # small pause to let things settle
    sleep 1
done

echo "Sweep complete. Results in $RESULTS_CSV"
