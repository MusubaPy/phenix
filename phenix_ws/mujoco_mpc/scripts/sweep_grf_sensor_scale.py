#!/usr/bin/env python3
"""Sweep a runtime GRF sensor scale (MJPC_GRF_SENSOR_SCALE) for chosen (hind,front,internal) combos.

Defaults: runs the three best combinations (from previous sweep) with scales in [1e-5 .. 1e-9]
with half-order steps (~9 values) producing ~27 runs.
"""
import argparse
import csv
import os
import subprocess
import sys
from math import log10

ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_BIN = os.environ.get('BUILD_BIN') or os.path.join(ROOT, 'build', 'bin', 'mjpc_mod')
OUTDIR = os.path.join(ROOT, 'logs', 'sweep_grf', 'sensor_scale')

# Top 3 combos (hind, front, internal) chosen from the previous sweep
DEFAULT_COMBOS = [
    (0.01, 0.03, 0.001),
    (0.01, 0.005, 1e-5),
    (0.0, 0.0, 0.0003)
]

# generate half-order-of-magnitude steps between 1e-5 and 1e-9
def generate_scales():
    vals = []
    # exponents from -5 to -9, include half-steps: -5, -5.5, -6, -6.5, ... -9
    e = -5.0
    while e >= -9.0 - 1e-9:
        vals.append(10 ** e)
        e -= 0.5
    return vals

SCALES = generate_scales()


def run_one(bin_path, outdir, hind, front, internal, scale, run_idx):
    tag = f"h{hind:g}_f{front:g}_i{internal:g}_s{scale:.0e}_r{run_idx}"
    csv_out = os.path.join(outdir, f"run_{tag}.csv")
    conv = csv_out.replace('.csv', '_conv.csv')
    env = os.environ.copy()
    env.update({
        'MJPC_CSV_LOG': csv_out,
        'MJPC_MAX_SIM_TIME': '45',
        'MJPC_GRF_HIND_WEIGHT': str(hind),
        'MJPC_GRF_FRONT_WEIGHT': str(front),
        'MJPC_INTERNAL_GRF_ALIGN_WEIGHT': str(internal),
        'MJPC_GRF_SENSOR_SCALE': str(scale),
        'MJPC_CTRL_CLIP': env.get('MJPC_CTRL_CLIP', '10'),
        'MJPC_MIN_TRAVEL_DISTANCE_M': env.get('MJPC_MIN_TRAVEL_DISTANCE_M','0'),
    })
    print(f"Run -> hind={hind} front={front} internal={internal} scale={scale} run={run_idx}")
    try:
        subprocess.run([bin_path, '--task=Quadruped Flat'], env=env, check=False)
    except Exception as e:
        print('ERROR running binary: ', e)
    status = 'OK' if os.path.isfile(csv_out) else 'MISSING'
    if status == 'OK':
        try:
            subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'convert_mjpc_csv.py'), csv_out, conv], check=True)
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

    return {'hind':hind,'front':front,'internal':internal,'scale':scale,'run':run_idx,'mp':mp,'mr':mr,'energy':en,'csv':conv if os.path.exists(conv) else csv_out,'status':status}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--combos', type=str, help='comma list of combos hind:front:internal e.g. 0.01:0.005:1e-4,...', default=None)
    p.add_argument('--run', action='store_true')
    p.add_argument('--bin', type=str, default=DEFAULT_BIN)
    p.add_argument('--outdir', type=str, default=OUTDIR)
    args = p.parse_args()

    # IMPORTANT: runtime GRF sensor scaling is disabled in the codebase as it
    # masks true energy effects. This sensor-scale sweep is therefore
    # intentionally unsupported. Exit early to avoid confusion.
    print('ERROR: MJPC_GRF_SENSOR_SCALE is disabled in the task code; sensor-scale sweeps are unsupported.')
    sys.exit(1)

    if args.combos:
        combos = []
        for s in args.combos.split(','):
            h,f,i = s.split(':')
            combos.append((float(h), float(f), float(i)))
    else:
        combos = DEFAULT_COMBOS

    print('Combos to test (count={}):'.format(len(combos)))
    for c in combos:
        print(' ', c)
    print('Scales (count={}):'.format(len(SCALES)), SCALES)
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
        writer.writerow(['hind','front','internal','scale','run','mp','mr','energy','csv','status'])
        for (h,fr,i_w) in combos:
            for s in SCALES:
                res = run_one(args.bin, args.outdir, h, fr, i_w, s, 1)
                writer.writerow([res['hind'],res['front'],res['internal'],res['scale'],res['run'],res['mp'],res['mr'],res['energy'],res['csv'],res['status']])
                out_f.flush()
                rows.append(res)

    print('All runs completed. Wrote ->', raw_csv)


if __name__ == '__main__':
    main()
