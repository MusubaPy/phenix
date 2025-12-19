#!/usr/bin/env python3
"""Sweep MJPC_TARGET_SMOOTH_TAU and MJPC_TARGET_BLEND_ALPHA combinations.
Runs the binary for 20s per combo, saves CSV and stdout log, then runs gait_evaluator to produce JSON and plots.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

# Default to the chosen single combination (tau=1.0, alpha=0.6) so a
# plain script invocation runs the one-off check without passing args.
DEFAULT_TAUS = [1.0]
DEFAULT_ALPHAS = [0.6]


def run_combo(bin_path, task, outdir, tau, alpha, extra_env=None, seed=None):
    odir = os.path.join(outdir, f'tau_{tau}_alpha_{alpha}')
    os.makedirs(odir, exist_ok=True)
    csv_path = os.path.join(odir, 'run.csv')
    log_path = os.path.join(odir, 'run.log')
    eval_json = os.path.join(odir, 'run_eval.json')
    plots_dir = os.path.join(odir, 'plots')

    env = os.environ.copy()
    # Explicitly set all relevant MJPC_* runtime knobs here so experiments
    # are fully reproducible and obvious (defaults chosen to preserve
    # current behavior). Feel free to override with --extra-env FOO=bar.
    env.update({
        # Core experiment controls
        'MJPC_CSV_LOG': csv_path,
        'MJPC_MAX_SIM_TIME': '60',           # seconds per run
        # Fix seed to 1 unless explicitly changed via --seed or --extra-env.
        # Do NOT inherit MJPC_SEED from the caller environment to avoid
        # accidental variation during quick checks.
        'MJPC_SEED': str(seed if seed is not None else 1),

        # Cost/penalty knobs
        'MJPC_GRF_WEIGHT': '1e-5',              # global GRF cost scalar (0 disables)
        'MJPC_GRF_PER_FOOT_SCALE': '0.8,0.8,1.2,1.2',  # per-foot GRF scaling
        'MJPC_GRF_HIND_WEIGHT': '1e-7',
        'MJPC_GRF_FRONT_WEIGHT': '1e-7',
        'MJPC_INTERNAL_GRF_ALIGN_WEIGHT': env.get('MJPC_INTERNAL_GRF_ALIGN_WEIGHT', '1e-3'),
        'MJPC_TARGET_SMOOTH_TAU': str(tau),  # smoothing tau
        'MJPC_TARGET_BLEND_ALPHA': str(alpha),
        'MJPC_FIXATION_WEIGHT': env.get('MJPC_FIXATION_WEIGHT', '1e-4'),
        'MJPC_POWER_PENALTY_WEIGHT': env.get('MJPC_POWER_PENALTY_WEIGHT', '0'),
        'MJPC_POWER_PENALTY_MODE': env.get('MJPC_POWER_PENALTY_MODE', ''),
        'MJPC_HEIGHT_WEIGHT_SCALE': env.get('MJPC_HEIGHT_WEIGHT_SCALE', '1.0'),
        'MJPC_BIARTICULAR_GAIN': env.get('MJPC_BIARTICULAR_GAIN', '0.0'),

        # GRF control/misc
        'MJPC_GRF_TRANSITION_BOOST': env.get('MJPC_GRF_TRANSITION_BOOST', '0.0'),
        'MJPC_GRF_NORMALIZE': env.get('MJPC_GRF_NORMALIZE', '0'),
        'MJPC_GRF_LOSS_MIX': env.get('MJPC_GRF_LOSS_MIX', '0.0'),
        'MJPC_GRF_MOTOR_BLEND': env.get('MJPC_GRF_MOTOR_BLEND', '0.0'),
        'MJPC_DISABLE_MOTOR_BLEND': env.get('MJPC_DISABLE_MOTOR_BLEND', '0'),
        'MJPC_GRF_TARGET_MODE': env.get('MJPC_GRF_TARGET_MODE', ''),

        # Simulation / stability controls
        'MJPC_CONTACT_STABLE_STEPS': env.get('MJPC_CONTACT_STABLE_STEPS', '10'),
        'MJPC_CTRL_CLIP': env.get('MJPC_CTRL_CLIP', '100'),
        'MJPC_MIN_TRAVEL_DISTANCE_M': env.get('MJPC_MIN_TRAVEL_DISTANCE_M', '0'),

        # Sensor / debugging (advanced)
        'MJPC_GRF_SENSOR_SCALE': env.get('MJPC_GRF_SENSOR_SCALE', '1.0'),
        'MJPC_REFLECT_NET_FORCE_GAIN': env.get('MJPC_REFLECT_NET_FORCE_GAIN', '0.0'),

        # Metrics (post-processing helpers)
        'MJPC_METRICS_START_SEC': env.get('MJPC_METRICS_START_SEC', ''),
        'MJPC_METRICS_END_SEC': env.get('MJPC_METRICS_END_SEC', ''),
    })
    if extra_env:
        env.update(extra_env)
    if seed is not None:
        env['MJPC_SEED'] = str(seed)

    cmd = [bin_path, '--task=' + task]
    start = datetime.now().isoformat()
    print(f'[{start}] Running: tau={tau} alpha={alpha} -> outdir={odir}')
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
    ap.add_argument('--outdir', required=False, default='logs/sweep_smooth', help='Output directory')
    ap.add_argument('--taus', nargs='*', type=float, default=DEFAULT_TAUS, help='List of taus')
    ap.add_argument('--alphas', nargs='*', type=float, default=DEFAULT_ALPHAS, help='List of alphas')
    ap.add_argument('--seed', type=int, default=1, help='Optional fixed MJPC_SEED (default: 1)')
    ap.add_argument('--extra-env', nargs='*', help='Extra ENV VARS like FOO=bar')
    args = ap.parse_args()

    extra_env = {}
    if args.extra_env:
        for kv in args.extra_env:
            if '=' in kv:
                k, v = kv.split('=', 1)
                extra_env[k] = v

    os.makedirs(args.outdir, exist_ok=True)

    for tau in args.taus:
        for alpha in args.alphas:
            run_combo(args.bin, args.task, args.outdir, tau, alpha, extra_env, seed=args.seed)

    print('All runs finished. Results are in', args.outdir)
