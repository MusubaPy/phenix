#!/usr/bin/env python3
"""Run an end-to-end baseline vs modified comparison with sensible defaults.

Usage: run without arguments. It will:
 - run 2 baseline (vanila) and 2 modified (mod) mjpc runs (using build/bin/mjpc_vanila and build/bin/mjpc_mod)
 - convert the mjpc CSV outputs via `convert_mjpc_csv.py`
 - trim converted CSVs to t in [10,60]s
 - compute metrics via `compute_metrics.py`
 - write `logs/baseline_vs_mod/results.csv` and print a short summary

The script is intentionally zero-config so you can run it without flags.
If you want to customize, edit the constants below or run `collect_datasets.py` for more control.
"""
from pathlib import Path
import subprocess
import os
import csv
import shlex
import sys
import statistics

# --- Configuration (sane defaults) ---
ROOT = Path(__file__).resolve().parents[1]
BIN_BASE = ROOT / 'build' / 'bin'
OUTDIR = ROOT / 'logs' / 'baseline_vs_mod'
RUNS_PER_TAG = 3
TMIN = 10.0
TMAX = 60.0
MAX_SIM_TIME = 62.0
TASK_NAME = 'Quadruped Flat'
TAGS = [('baseline', 'mjpc_vanila'), ('modified', 'mjpc_mod')]

# Per-tag environment overrides (keep baseline unchanged; use conservative params for mod)
TAG_ENV_OVERRIDES = {
    'baseline': {},
    'modified': {
        'MJPC_CTRL_CLIP': '50',
        'MJPC_GRF_HIND_WEIGHT': '0.03',
        'MJPC_GRF_FRONT_WEIGHT': '0.01',
        # ensure a slightly longer sim window (safe) if needed
        'MJPC_MAX_SIM_TIME': str(MAX_SIM_TIME),
    }
}


def run_once(binpath, task, out_csv, env_overrides=None, max_sim_time=None):
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    env['MJPC_CSV_LOG'] = str(out_csv)
    if max_sim_time is not None:
        env['MJPC_MAX_SIM_TIME'] = str(max_sim_time)
    cmd = [str(binpath), '--task=' + task]
    print('Running', ' '.join(shlex.quote(c) for c in cmd), '->', out_csv)
    p = subprocess.run(cmd, input='\n', text=True, env=env)
    return p.returncode == 0 and Path(out_csv).exists()


def convert_and_trim(conv_script_dir, csv_path, tmin, tmax):
    conv = Path(str(csv_path).replace('.csv', '_conv.csv'))
    subprocess.run(['python3', str(conv_script_dir / 'convert_mjpc_csv.py'), str(csv_path), str(conv)], check=True)
    # trim
    trimmed = Path(str(conv).replace('.csv', f'_trim_{int(tmin)}_{int(tmax)}.csv'))
    with conv.open(newline='') as inf, trimmed.open('w', newline='') as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)
        try:
            header = next(reader)
        except StopIteration:
            return conv, None
        writer.writerow(header)
        for row in reader:
            if not row:
                continue
            try:
                t = float(row[0])
            except Exception:
                continue
            if (tmin is None or t >= tmin) and (tmax is None or t <= tmax):
                writer.writerow(row)
    return conv, trimmed


def compute_metrics(conv_path):
    p = subprocess.run(['python3', str(Path(__file__).resolve().parent / 'compute_metrics.py'), str(conv_path)], capture_output=True, text=True)
    return p


def extract_distance(conv_path):
    try:
        with open(conv_path, newline='') as f:
            last = list(csv.reader(f))[-1]
            return last[4] if len(last) > 4 else 'NA'
    except Exception:
        return 'NA'


def summarize_results(results_csv):
    rows = []
    with open(results_csv, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    out = {}
    for tag in set(r['tag'] for r in rows):
        s = [x for x in rows if x['tag'] == tag]
        mp = [float(x['margin_pitch_mean']) for x in s if x['margin_pitch_mean'] not in ('NA', '')]
        mr = [float(x['margin_roll_mean']) for x in s if x['margin_roll_mean'] not in ('NA', '')]
        e = [float(x['energy_abs_mean']) for x in s if x['energy_abs_mean'] not in ('NA', '')]
        out[tag] = {
            'n': len(s),
            'pitch_mean': statistics.mean(mp) if mp else None,
            'pitch_std': statistics.pstdev(mp) if len(mp) > 1 else 0.0,
            'roll_mean': statistics.mean(mr) if mr else None,
            'roll_std': statistics.pstdev(mr) if len(mr) > 1 else 0.0,
            'energy_mean': statistics.mean(e) if e else None,
            'energy_std': statistics.pstdev(e) if len(e) > 1 else 0.0,
        }
    return out


def main():
    import shutil
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results_csv = OUTDIR / 'results.csv'
    with results_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag','run','margin_pitch_mean','margin_roll_mean','energy_abs_mean','csv','distance_traveled_m','status'])

    conv_script_dir = Path(__file__).resolve().parent

    for tag, binname in TAGS:
        binpath = BIN_BASE / binname
        if not binpath.exists():
            print(f'Warning: expected binary {binpath} not found. Trying PATH lookup...')
            binpath = shutil.which(binname)
            if not binpath:
                print('ERROR: binary for', tag, 'not found, skipping.')
                continue
        for i in range(1, RUNS_PER_TAG + 1):
            csv_path = OUTDIR / f"{tag}_r{i}.csv"
            if csv_path.exists():
                csv_path.unlink()
            env_overrides = TAG_ENV_OVERRIDES.get(tag, {})
            ok = run_once(binpath, TASK_NAME, csv_path, env_overrides=env_overrides, max_sim_time=MAX_SIM_TIME)
            if not csv_path.exists():
                print('ERROR: missing', csv_path)
                with results_csv.open('a', newline='') as f:
                    csv.writer(f).writerow([tag, i, 'NA','NA','NA', str(csv_path), 'NA', 'ERROR'])
                continue
            try:
                conv, trimmed = convert_and_trim(conv_script_dir, csv_path, TMIN, TMAX)
            except subprocess.CalledProcessError:
                print('Conversion failed for', csv_path)
                with results_csv.open('a', newline='') as f:
                    csv.writer(f).writerow([tag, i, 'NA','NA','NA', str(csv_path), 'NA', 'CONVERT_FAIL'])
                continue

            if trimmed is None:
                print('Trimming produced no data for', conv)
                with results_csv.open('a', newline='') as f:
                    csv.writer(f).writerow([tag, i, 'NA','NA','NA', str(conv), 'NA', 'NO_DATA'])
                continue

            res = compute_metrics(trimmed)
            stdout = res.stdout.strip()
            if res.returncode == 2 or stdout == 'NO_DATA':
                status = 'NO_DATA'
                with results_csv.open('a', newline='') as f:
                    csv.writer(f).writerow([tag, i, 'NA','NA','NA', str(trimmed), 'NA', status])
                continue
            if res.returncode == 3 or stdout.startswith('NO_TRAVEL'):
                dist = extract_distance(trimmed)
                status = 'NO_TRAVEL'
                with results_csv.open('a', newline='') as f:
                    csv.writer(f).writerow([tag, i, 'NA','NA','NA', str(trimmed), dist, status])
                continue
            if res.returncode != 0:
                print('compute_metrics returned non-zero:', res.returncode, res.stderr)
                status = 'METRIC_FAIL'
                with results_csv.open('a', newline='') as f:
                    csv.writer(f).writerow([tag, i, 'NA','NA','NA', str(trimmed), 'NA', status])
                continue

            mp, mr, en = stdout.split(',')
            dist = extract_distance(trimmed)
            with results_csv.open('a', newline='') as f:
                csv.writer(f).writerow([tag, i, mp, mr, en, str(trimmed), dist, 'OK'])

    # final summary
    summary = summarize_results(results_csv)
    print('\nSummary:')
    for tag, stats in summary.items():
        print(f"{tag}: n={stats['n']}, pitch={stats['pitch_mean']} (±{stats['pitch_std']}), roll={stats['roll_mean']} (±{stats['roll_std']}), energy={stats['energy_mean']} (±{stats['energy_std']})")
    print('\nDone. Results:', results_csv)


if __name__ == '__main__':
    main()
