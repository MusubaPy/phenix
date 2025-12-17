#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BUILD_BIN:-$ROOT_DIR/build/bin/mjpc}"
OUTDIR="$ROOT_DIR/logs/sweep_grf/baseline_vs_mod"
mkdir -p "$OUTDIR"

WEIGHT=${1:-1e-8}
N=${2:-5}

RESULTS_CSV="$OUTDIR/results.csv"
SUMMARY_CSV="$OUTDIR/summary.csv"
rm -f "$RESULTS_CSV" "$SUMMARY_CSV"
echo "weight,run,margin_pitch_mean,margin_roll_mean,energy_abs_mean,csv,distance_traveled_m,status" > "$RESULTS_CSV"

run_one() {
    local w=$1
    local idx=$2
    local tag=$3
    local csv="$OUTDIR/${tag}_w$(echo "$w" | sed 's/\./p/g; s/\-//g; s/1e/1e/g')_r${idx}.csv"
    local conv="${csv%.csv}_conv.csv"

    echo "Running $tag weight=$w run=$idx -> $csv"
    MJPC_CSV_LOG="$csv" MJPC_MAX_SIM_TIME=45 MJPC_GRF_WEIGHT="$w" "$BIN" --task="Quadruped Flat" || true
    if [[ ! -f "$csv" ]]; then
        echo "ERROR: missing $csv" >&2
        echo "$w,$idx,NA,NA,NA,$csv,NA,ERROR" >> "$RESULTS_CSV"
        return
    fi
    python3 "$ROOT_DIR/scripts/convert_mjpc_csv.py" "$csv" "$conv"
    out=$(python3 "$ROOT_DIR/scripts/compute_metrics.py" "$conv" 2>&1 || true)
    if [[ "$out" == "NO_DATA" ]]; then
        echo "NO_DATA"
        echo "$w,$idx,NA,NA,NA,$conv,NA,NO_DATA" >> "$RESULTS_CSV"
        return
    fi
    if [[ "$out" == NO_TRAVEL* ]]; then
        # extract final distance if available
        dist=$(tail -n 1 "$conv" | awk -F',' '{print $5}' 2>/dev/null || echo NA)
        echo "NO_TRAVEL"
        echo "$w,$idx,NA,NA,NA,$conv,$dist,NO_TRAVEL" >> "$RESULTS_CSV"
        return
    fi
    mp=$(echo "$out" | cut -d',' -f1)
    mr=$(echo "$out" | cut -d',' -f2)
    en=$(echo "$out" | cut -d',' -f3)
    dist=$(tail -n 1 "$conv" | awk -F',' '{print $5}' 2>/dev/null || echo NA)
    echo "$w,$idx,$mp,$mr,$en,$conv,$dist,OK" >> "$RESULTS_CSV"
}

echo "=== Baseline runs (no GRF cost) ==="
for i in $(seq 1 $N); do
    run_one 0 $i baseline
done

echo "=== Modified runs (weight=$WEIGHT) ==="
for i in $(seq 1 $N); do
    run_one "$WEIGHT" $i modified
done

# compute summary stats
python3 - <<PY
import pandas as pd
df = pd.read_csv('$RESULTS_CSV')
def summarize(tag):
    sub = df[(df['weight']==0) & (df['csv'].str.contains('baseline'))] if tag=='baseline' else df[(df['weight']==float('$WEIGHT')) & (df['csv'].str.contains('modified'))]
    ok = sub[sub['status']=='OK']
    def stats(col):
        s = ok[col].dropna().astype(float)
        if s.empty:
            return pd.Series({'mean':pd.NA,'std':pd.NA,'n':0})
        return pd.Series({'mean':s.mean(),'std':s.std(),'n':len(s)})
    pm = stats('margin_pitch_mean')
    rm = stats('margin_roll_mean')
    em = stats('energy_abs_mean')
    out = {'tag':tag,'pitch_mean':pm['mean'],'pitch_std':pm['std'],'pitch_n':pm['n'],'roll_mean':rm['mean'],'roll_std':rm['std'],'roll_n':rm['n'],'energy_mean':em['mean'],'energy_std':em['std'],'energy_n':em['n']}
    print(out)
    return out

base = summarize('baseline')
mod = summarize('modified')
import csv
with open('$SUMMARY_CSV','w') as f:
    w = csv.writer(f)
    w.writerow(['metric','baseline_mean','baseline_std','baseline_n','modified_mean','modified_std','modified_n','delta_mean'])
    def wr(row, col):
        b = row[col+'_mean'] if row[col+'_mean'] is not pd.NA else 'NA'
        bs = row[col+'_std'] if row[col+'_std'] is not pd.NA else 'NA'
        bn = int(row[col+'_n']) if row[col+'_n'] is not pd.NA else 0
        m = mod[col+'_mean'] if mod[col+'_mean'] is not pd.NA else 'NA'
        ms = mod[col+'_std'] if mod[col+'_std'] is not pd.NA else 'NA'
        mn = int(mod[col+'_n']) if mod[col+'_n'] is not pd.NA else 0
        delta = 'NA'
        try:
            if b!='NA' and m!='NA':
                delta = float(m) - float(b)
        except Exception:
            delta = 'NA'
        return [col,b,bs,bn,m,ms,mn,delta]
    for col in ['pitch','roll','energy']:
        w.writerow(wr(base,col))
print('Wrote summary -> $SUMMARY_CSV')
PY

echo "Done. Results: $RESULTS_CSV; Summary: $SUMMARY_CSV"
