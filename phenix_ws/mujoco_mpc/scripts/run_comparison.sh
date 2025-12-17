#!/usr/bin/env bash
set -euo pipefail

# Simple runner for two mjpc experiments (with/without internal GRF alignment)
# It runs mjpc twice (max sim time 45s), converts the outputs and runs comparison.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_BIN="$ROOT_DIR/build/bin/mjpc"
LOG_DIR="$ROOT_DIR/logs/compare_runs"
mkdir -p "$LOG_DIR"

WITH_CSV="$LOG_DIR/run_with.csv"
WITHOUT_CSV="$LOG_DIR/run_without.csv"
WITH_CONV="$LOG_DIR/run_with_converted.csv"
WITHOUT_CONV="$LOG_DIR/run_without_converted.csv"

echo "Running mjpc WITH internal GRF alignment..."
MJPC_CSV_LOG="$WITH_CSV" MJPC_MAX_SIM_TIME=45 MJPC_INTERNAL_GRF_ALIGN_WEIGHT=0.001 "$BUILD_BIN" --task="Quadruped Flat"
if [[ ! -f "$WITH_CSV" ]]; then
	echo "ERROR: expected CSV $WITH_CSV not found" >&2
	exit 1
fi

echo "Running mjpc WITHOUT internal GRF alignment..."
MJPC_CSV_LOG="$WITHOUT_CSV" MJPC_MAX_SIM_TIME=45 MJPC_INTERNAL_GRF_ALIGN_WEIGHT=0.0 "$BUILD_BIN" --task="Quadruped Flat"
if [[ ! -f "$WITHOUT_CSV" ]]; then
	echo "ERROR: expected CSV $WITHOUT_CSV not found" >&2
	exit 1
fi

echo "Converting CSVs to dataset format..."
python3 "$ROOT_DIR/scripts/convert_mjpc_csv.py" "$WITH_CSV" "$WITH_CONV"
python3 "$ROOT_DIR/scripts/convert_mjpc_csv.py" "$WITHOUT_CSV" "$WITHOUT_CONV"

echo "Comparing runs..."
python3 "$ROOT_DIR/scripts/compare_runs.py" "$WITH_CONV" "$WITHOUT_CONV"

echo "Done. Logs and converted datasets are in: $LOG_DIR"
