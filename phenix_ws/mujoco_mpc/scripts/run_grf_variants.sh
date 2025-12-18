#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BUILD_BIN:-$ROOT_DIR/build/bin/mjpc}"
OUTDIR="$ROOT_DIR/logs/sweep_grf/variants"
mkdir -p "$OUTDIR"

WEIGHT=${1:-1e-8}
N=${2:-5}

# Variant definitions
# A: per-foot scaling (boost hind legs slightly)
VAR_A_ENV="MJPC_GRF_PER_FOOT_SCALE=1.2,1.2,1.0,1.0"
# B: normalization + L1/L2 mix
VAR_B_ENV="MJPC_GRF_NORMALIZE=1 MJPC_GRF_LOSS_MIX=0.5"
# C: transition scheduler boost
VAR_C_ENV="MJPC_GRF_TRANSITION_BOOST=2.0"

run_variant() {
    local tag=$1
    local envs=$2
    echo "=== Variant $tag ($envs) ==="
    for i in $(seq 1 $N); do
        csv="$OUTDIR/${tag}_w$(echo "$WEIGHT" | sed 's/\./p/g; s/\-//g; s/1e/1e/g')_r${i}.csv"
        conv="${csv%.csv}_conv.csv"
        echo "Running $tag run=$i -> $csv"
        # shellcheck disable=SC2086
        # pipe a newline to avoid interactive "Press Enter to exit" pauses
        echo '' | env $envs MJPC_CSV_LOG="$csv" MJPC_MAX_SIM_TIME=45 MJPC_GRF_WEIGHT="$WEIGHT" "$BIN" --task="Quadruped Flat" || true
        if [[ ! -f "$csv" ]]; then
            echo "Missing $csv"
            continue
        fi
        python3 "$ROOT_DIR/scripts/convert_mjpc_csv.py" "$csv" "$conv"
        python3 "$ROOT_DIR/scripts/compute_metrics.py" "$conv" || true
    done
}

run_variant "A_perfoot" "$VAR_A_ENV"
run_variant "B_norm_mix" "$VAR_B_ENV"
run_variant "C_transition" "$VAR_C_ENV"

echo "Done. Outputs in $OUTDIR"