#!/usr/bin/env bash
set -euo pipefail

# Rerun missing sweep combos for a given sweep OUTDIR (defaults to last created)
OUTDIR=${1:-$(ls -dt logs/sweep_alex_* 2>/dev/null | head -1)}
if [[ -z "$OUTDIR" ]]; then
  echo "No sweep output dir found. Pass OUTDIR as first arg." >&2
  exit 1
fi

MJPC_BIN="$(pwd)/build/bin/mjpc_mod"
if [[ ! -x "$MJPC_BIN" ]]; then
  MJPC_BIN="build/bin/mjpc_mod"
fi

SIMTIME=${SIMTIME:-30}
FX_VALUES=(1e-9 1e-10)

cat > "$OUTDIR/rerun_missing_commands.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
OUTDIR="__OUTDIR__"
MJPC_BIN="__MJPC_BIN__"
SIMTIME=__SIMTIME__

for pe in $(seq 1 10); do
  p=$(printf "1e-%d" "$pe")
  for ae in $(seq 1 10); do
    a=$(printf "1e-%d" "$ae")
    for fx in 1e-9 1e-10; do
      for r in 1 2 3; do
        name=$(printf "p%s_a%s_fx%s_r%d" "$p" "$a" "$fx" "$r")
        csv="$OUTDIR/${name}.csv"
        if [[ -s "$csv" ]]; then
          continue
        fi
        echo "Running missing: $name -> $csv"
        xvfb-run -s "-screen 0 1280x1024x24" env MJPC_CSV_LOG="$csv" timeout ${SIMTIME}s "$MJPC_BIN" \
          --task "Quadruped Flat (mod)" \
          --alex_enabled \
          --alex_power_weight=${p} \
          --alex_align_weight=${a} \
          --alex_fx_smooth_weight=${fx} || true
        if [[ ! -s "$csv" ]]; then
          echo "$(date +%s) $p $a $fx $r" >> "$OUTDIR/failed_runs.txt"
          echo "[WARN] Run failed or CSV missing: $csv"
        else
          echo "$csv" >> "$OUTDIR/completed_runs.txt"
        fi
      done
    done
  done
done
SH

# substitute placeholders
sed -i "s|__OUTDIR__|$OUTDIR|g" "$OUTDIR/rerun_missing_commands.sh"
sed -i "s|__MJPC_BIN__|$MJPC_BIN|g" "$OUTDIR/rerun_missing_commands.sh"
sed -i "s|__SIMTIME__|$SIMTIME|g" "$OUTDIR/rerun_missing_commands.sh"

chmod +x "$OUTDIR/rerun_missing_commands.sh"

echo "Created $OUTDIR/rerun_missing_commands.sh"
echo "Inspect it, run it to re-run missing combos (it will append to failed_runs.txt and completed_runs.txt)."

echo "Suggested smoke run (first 1 command):"
head -n 40 "$OUTDIR/rerun_missing_commands.sh" | sed -n '1,40p' 

