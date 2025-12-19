#!/usr/bin/env python3
"""Run a grid sweep over GRF weights and aggregate metrics.

Usage (examples):
  python3 scripts/sweep_grf_grid.py --hind 0.0,0.003,0.01 --front 0.0,0.001,0.005 --internal 0.0,1e-4 --n 3

This script:
 - launches the mjpc binary for each (hind,front,internal) triple for N seeds
 - converts CSVs and runs `compute_metrics.py` for each run
 - writes raw results to logs/sweep_grf/grid/raw_results.csv
 - writes aggregated summary to logs/sweep_grf/grid/summary.csv
 - ranks combos by (pitch_mean + roll_mean) (higher is better) then lower energy
"""

import argparse
import csv
import os
import subprocess
import sys
import math
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_BIN = os.environ.get('BUILD_BIN') or os.path.join(ROOT, 'build', 'bin', 'mjpc_mod')


def parse_list(s):
    return [float(x) for x in s.split(',') if x.strip()!='']


def run_one(bin_path, outdir, hind, front, internal, run_idx, max_sim=45, extra_env=None):
    tag = f"h{hind:g}_f{front:g}_i{internal:g}_r{run_idx}"
    csv = os.path.join(outdir, f"run_{tag}.csv")
    conv = csv.replace('.csv', '_conv.csv')
    env = os.environ.copy()
    env.update({
        'MJPC_CSV_LOG': csv,
        'MJPC_MAX_SIM_TIME': str(max_sim),
        'MJPC_GRF_HIND_WEIGHT': str(hind),
        'MJPC_GRF_FRONT_WEIGHT': str(front),
        'MJPC_INTERNAL_GRF_ALIGN_WEIGHT': str(internal),
        # conservative control clipping to reduce NaNs
        'MJPC_CTRL_CLIP': env.get('MJPC_CTRL_CLIP','50')
    })
    if extra_env:
        env.update(extra_env)
    print(f"Running: hind={hind} front={front} internal={internal} run={run_idx}")
    try:
        subprocess.run([bin_path, '--task=Quadruped Flat'], env=env, check=False)
    except Exception as e:
        print("Run failed:", e)
    # verify csv
    status = 'OK' if os.path.isfile(csv) else 'MISSING'
    if status == 'OK':
        # convert
        try:
            subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'convert_mjpc_csv.py'), csv, conv], check=True)
        except Exception as e:
            print('convert failed', e)
            status = 'CONVERT_FAIL'
    # compute metrics
    mp = mr = en = 'NA'
    if status == 'OK':
        try:
            out = subprocess.check_output([sys.executable, os.path.join(ROOT, 'scripts', 'compute_metrics.py'), conv], stderr=subprocess.STDOUT)
            out = out.decode().strip()
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

    return {
        'hind': hind,
        'front': front,
        'internal': internal,
        'run': run_idx,
        'mp': mp,
        'mr': mr,
        'energy': en,
        'csv': conv if os.path.exists(conv) else (csv if os.path.exists(csv) else ''),
        'status': status
    }


def aggregate(rows, outdir):
    # group by (hind,front,internal)
    groups = defaultdict(list)
    for r in rows:
        key = (r['hind'], r['front'], r['internal'])
        groups[key].append(r)

    summary_rows = []
    for key, runs in groups.items():
        hind, front, internal = key
        mp_vals = [float(r['mp']) for r in runs if r['mp']!='NA' and r['status']=='OK']
        mr_vals = [float(r['mr']) for r in runs if r['mr']!='NA' and r['status']=='OK']
        en_vals = [float(r['energy']) for r in runs if r['energy']!='NA' and r['status']=='OK']
        def stats(v):
            if not v:
                return ('NA','NA',0)
            import statistics
            return (statistics.mean(v), statistics.pstdev(v) if len(v)>1 else 0.0, len(v))
        mp_mean, mp_std, mp_n = stats(mp_vals)
        mr_mean, mr_std, mr_n = stats(mr_vals)
        en_mean, en_std, en_n = stats(en_vals)
        score = 'NA'
        if mp_mean!='NA' and mr_mean!='NA':
            score = (mp_mean + mr_mean)
        summary_rows.append({'hind':hind,'front':front,'internal':internal,'pitch_mean':mp_mean,'pitch_std':mp_std,'pitch_n':mp_n,'roll_mean':mr_mean,'roll_std':mr_std,'roll_n':mr_n,'energy_mean':en_mean,'energy_std':en_std,'energy_n':en_n,'score':score})

    # write summary CSV
    summary_csv = os.path.join(outdir, 'summary.csv')
    keys = ['hind','front','internal','pitch_mean','pitch_std','pitch_n','roll_mean','roll_std','roll_n','energy_mean','energy_std','energy_n','score']
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        # sort by score desc then energy asc
        def sort_key(r):
            s = r['score'] if r['score']!='NA' else -math.inf
            em = r['energy_mean'] if r['energy_mean']!='NA' else math.inf
            return (-s, em)
        for r in sorted(summary_rows, key=sort_key):
            w.writerow(r)

    print('Wrote summary ->', summary_csv)
    return summary_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hind', type=str, default='0.0,0.003,0.01', help='comma list of hind weights')
    p.add_argument('--front', type=str, default='0.0,0.001,0.005', help='comma list of front weights')
    p.add_argument('--internal', type=str, default='0.0,1e-4,1e-3', help='comma list of internal weights')
    p.add_argument('-n','--n', type=int, default=3, help='runs per combo')
    p.add_argument('--bin', type=str, default=DEFAULT_BIN, help='path to mjpc binary')
    p.add_argument('--outdir', type=str, default=os.path.join(ROOT, 'logs', 'sweep_grf', 'grid'), help='output dir')
    args = p.parse_args()

    hind_list = parse_list(args.hind)
    front_list = parse_list(args.front)
    internal_list = parse_list(args.internal)

    os.makedirs(args.outdir, exist_ok=True)
    raw_csv = os.path.join(args.outdir, 'raw_results.csv')
    rows = []
    with open(raw_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hind','front','internal','run','mp','mr','energy','csv','status'])
        for hind in hind_list:
            for front in front_list:
                for internal in internal_list:
                    for run_idx in range(1, args.n+1):
                        res = run_one(args.bin, args.outdir, hind, front, internal, run_idx)
                        writer.writerow([res['hind'],res['front'],res['internal'],res['run'],res['mp'],res['mr'],res['energy'],res['csv'],res['status']])
                        f.flush()
                        rows.append(res)

    print('All runs done. Aggregate...')
    aggregate(rows, args.outdir)


if __name__ == '__main__':
    main()
