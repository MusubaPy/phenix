#!/usr/bin/env python3
"""Run sweeps over a *fixed* list of (hind,front) pairs and a list of internal weights.

Default behavior: 10 pre-chosen hind/front pairs × 10 internal weights, 1 run each.

Outputs:
 - logs/sweep_grf/pairs/raw_results.csv
 - logs/sweep_grf/pairs/summary.csv

Use --run to actually execute; otherwise the script will print the plan.
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
import math

ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_BIN = os.environ.get('BUILD_BIN') or os.path.join(ROOT, 'build', 'bin', 'mjpc_mod')


DEFAULT_PAIRS = [
    (0.0, 0.0),
    (0.01, 0.005),
    (0.03, 0.01),
    (0.06, 0.03),
    (0.03, 0.003),
    (0.01, 0.01),
    (0.02, 0.005),
    (0.04, 0.015),
    (0.005, 0.001),
    (0.01, 0.03),  # front-heavy case
]

DEFAULT_INTERNAL = [0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def run_one(bin_path, outdir, hind, front, internal, run_idx):
    tag = f"h{hind:g}_f{front:g}_i{internal:g}_r{run_idx}"
    csv = os.path.join(outdir, f"run_{tag}.csv")
    conv = csv.replace('.csv', '_conv.csv')
    env = os.environ.copy()
    env.update({
        'MJPC_CSV_LOG': csv,
        'MJPC_MAX_SIM_TIME': '45',
        'MJPC_GRF_HIND_WEIGHT': str(hind),
        'MJPC_GRF_FRONT_WEIGHT': str(front),
        'MJPC_INTERNAL_GRF_ALIGN_WEIGHT': str(internal),
        'MJPC_CTRL_CLIP': env.get('MJPC_CTRL_CLIP', '50'),
    })
    print(f"Run -> hind={hind} front={front} internal={internal} run={run_idx}")
    try:
        subprocess.run([bin_path, '--task=Quadruped Flat'], env=env, check=False)
    except Exception as e:
        print('ERROR running binary: ', e)
    status = 'OK' if os.path.isfile(csv) else 'MISSING'
    if status == 'OK':
        try:
            subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'convert_mjpc_csv.py'), csv, conv], check=True)
        except Exception as e:
            print('convert failed', e)
            status = 'CONVERT_FAIL'
    mp = mr = en = 'NA'
    if status == 'OK':
        try:
            out = subprocess.check_output([sys.executable, os.path.join(ROOT, 'scripts', 'compute_metrics.py'), conv], stderr=subprocess.STDOUT).decode().strip()
            if out in ('NO_DATA','NO_TRAVEL'):
                status = out
            else:
                parts = out.split(',')
                if len(parts) >= 3:
                    mp, mr, en = parts[0], parts[1], parts[2]
                else:
                    status = 'METRICS_PARSE_FAIL'
        except subprocess.CalledProcessError as e:
            print('compute_metrics failed:', e.output.decode() if e.output else e)
            status = 'METRICS_FAIL'
        except Exception as e:
            print('compute_metrics exception', e)
            status = 'METRICS_FAIL'

    return {'hind':hind,'front':front,'internal':internal,'run':run_idx,'mp':mp,'mr':mr,'energy':en,'csv':conv if os.path.exists(conv) else csv,'status':status}


def aggregate(rows, outdir):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['hind'], r['front'])].append(r)

    summary = []
    for key, runs in groups.items():
        hind, front = key
        # aggregate across internal weights
        def collect(field):
            vals = [float(r[field]) for r in runs if r[field] != 'NA' and r['status']=='OK']
            return vals
        pitch_vals = collect('mp')
        roll_vals = collect('mr')
        energy_vals = collect('energy')
        def stats(v):
            if not v: return ('NA','NA',0)
            import statistics
            return (statistics.mean(v), statistics.pstdev(v) if len(v)>1 else 0.0, len(v))
        pm, psd, pn = stats(pitch_vals)
        rm, rsd, rn = stats(roll_vals)
        em, esd, en = stats(energy_vals)
        score = 'NA' if pm=='NA' or rm=='NA' else (pm + rm)
        summary.append({'hind':hind,'front':front,'pitch_mean':pm,'pitch_std':psd,'pitch_n':pn,'roll_mean':rm,'roll_std':rsd,'roll_n':rn,'energy_mean':em,'energy_std':esd,'energy_n':en,'score':score})

    summary_csv = os.path.join(outdir, 'summary.csv')
    keys = ['hind','front','pitch_mean','pitch_std','pitch_n','roll_mean','roll_std','roll_n','energy_mean','energy_std','energy_n','score']
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        def sort_key(r):
            s = r['score'] if r['score']!='NA' else -math.inf
            em = r['energy_mean'] if r['energy_mean']!='NA' else math.inf
            return (-s, em)
        for r in sorted(summary, key=sort_key):
            w.writerow(r)

    print('Wrote summary ->', summary_csv)
    return summary_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', type=str, help='comma list of pairs hind:front e.g. 0.01:0.005,0.03:0.01', default=None)
    p.add_argument('--internal', type=str, help='comma list of internal weights', default=None)
    p.add_argument('--run', action='store_true', help='actually run the experiments; otherwise print plan')
    p.add_argument('--bin', type=str, default=DEFAULT_BIN)
    p.add_argument('--outdir', type=str, default=os.path.join(ROOT, 'logs', 'sweep_grf', 'pairs'))
    args = p.parse_args()

    if args.pairs:
        pairs = []
        for s in args.pairs.split(','):
            h,f = s.split(':')
            pairs.append((float(h), float(f)))
    else:
        pairs = DEFAULT_PAIRS

    if args.internal:
        internal = [float(x) for x in args.internal.split(',')]
    else:
        internal = DEFAULT_INTERNAL

    print('Pairs to test (count={}):'.format(len(pairs)))
    for p_ in pairs:
        print(' ', p_)
    print('Internal weights (count={}):'.format(len(internal)), internal)
    print('Runs per combo: 1 (single seed)')
    print('Binary:', args.bin)
    print('Outdir:', args.outdir)

    if not args.run:
        print('\nDry run. Use --run to execute.')
        return

    os.makedirs(args.outdir, exist_ok=True)
    raw_csv = os.path.join(args.outdir, 'raw_results.csv')
    rows = []
    with open(raw_csv, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['hind','front','internal','run','mp','mr','energy','csv','status'])
        for (h,fr) in pairs:
            for i_w in internal:
                res = run_one(args.bin, args.outdir, h, fr, i_w, 1)
                writer.writerow([res['hind'],res['front'],res['internal'],res['run'],res['mp'],res['mr'],res['energy'],res['csv'],res['status']])
                out_f.flush()
                rows.append(res)

    print('All runs completed. Aggregating per (hind,front) over internal weights...')
    aggregate(rows, args.outdir)


if __name__ == '__main__':
    main()
