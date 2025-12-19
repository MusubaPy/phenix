#!/usr/bin/env python3
"""Sweep fixation × power_penalty × height_weight_scale combinations.
Runs the binary for 20s per combo, saves CSV and stdout log, then runs gait_evaluator to produce JSON and plots.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

DEFAULT_FIX = [0.02, 0.05, 0.1, 0.2]
DEFAULT_POWER = [0.0, 0.001, 0.01]
DEFAULT_HEIGHT = [1.0, 1.5, 2.0]


def run_combo(bin_path, task, outdir, fix_w, power_w, height_scale, seed=None, extra_env=None):
    odir = os.path.join(outdir, f'fix_{fix_w}_pow_{power_w}_h_{height_scale}')
    os.makedirs(odir, exist_ok=True)
    csv_path = os.path.join(odir, 'run.csv')
    log_path = os.path.join(odir, 'run.log')
    eval_json = os.path.join(odir, 'run_eval.json')
    plots_dir = os.path.join(odir, 'plots')

    env = os.environ.copy()
    env.update({
        'MJPC_GRF_WEIGHT': '0',
        'MJPC_FIXATION_WEIGHT': str(fix_w),
        'MJPC_POWER_PENALTY_WEIGHT': str(power_w),
        'MJPC_HEIGHT_WEIGHT_SCALE': str(height_scale),
        'MJPC_TARGET_SMOOTH_TAU': '1.0',
        'MJPC_TARGET_BLEND_ALPHA': '0.6',
        'MJPC_CONTACT_STABLE_STEPS': '10',
        'MJPC_MAX_SIM_TIME': '20',
        'MJPC_CSV_LOG': csv_path,
    })
    if seed is not None:
        env['MJPC_SEED'] = str(seed)
    if extra_env:
        env.update(extra_env)

    cmd = [bin_path, '--task=' + task]
    start = datetime.now().isoformat()
    print(f'[{start}] Running: fix={fix_w} power={power_w} height={height_scale} -> outdir={odir}')
    with open(log_path, 'wb') as outf:
        proc = subprocess.run(cmd, env=env, stdout=outf, stderr=subprocess.STDOUT)
    end = datetime.now().isoformat()
    print(f'[{end}] Finished (returncode={proc.returncode})')

    # run evaluator
    eval_cmd = [sys.executable, 'scripts/gait_evaluator.py', '--csv', csv_path, '--out', eval_json, '--plots', plots_dir]
    print('Running evaluator:', ' '.join(eval_cmd))
    subprocess.run(eval_cmd)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=False, default='build/bin/mjpc_mod', help='Path to mjpc binary')
    ap.add_argument('--task', required=False, default='Quadruped Flat (mod)', help='Task name')
    ap.add_argument('--outdir', required=False, default='logs/sweep_weights', help='Output directory')
    ap.add_argument('--fix', nargs='*', type=float, default=DEFAULT_FIX, help='List of fixation weights')
    ap.add_argument('--power', nargs='*', type=float, default=DEFAULT_POWER, help='List of power penalty weights')
    ap.add_argument('--height', nargs='*', type=float, default=DEFAULT_HEIGHT, help='List of height scales')
    ap.add_argument('--seed', type=int, default=1, help='Optional fixed seed to use for all runs (default: 1)')
    ap.add_argument('--extra-env', nargs='*', help='Extra ENV VARS like FOO=bar')
    args = ap.parse_args()

    extra_env = {}
    if args.extra_env:
        for kv in args.extra_env:
            if '=' in kv:
                k, v = kv.split('=', 1)
                extra_env[k] = v

    os.makedirs(args.outdir, exist_ok=True)

    for f in args.fix:
        for p in args.power:
            for h in args.height:
                run_combo(args.bin, args.task, args.outdir, f, p, h, seed=args.seed, extra_env=extra_env)

    print('All runs finished. Results are in', args.outdir)
