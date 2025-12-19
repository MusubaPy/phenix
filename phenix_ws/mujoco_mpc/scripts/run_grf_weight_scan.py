#!/usr/bin/env python3
"""Run scans over a set of GRF weights (sets both hind and front weights equal)
for a list of chosen (hind,front,internal) combos.

Default behavior: run the top-3 energy combos with grf weights [1e-5..1e-9]
(one run per combo x weight).
"""
import argparse
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_BIN = os.environ.get('BUILD_BIN') or os.path.join(ROOT, 'build', 'bin', 'mjpc_mod')
OUTDIR = os.path.join(ROOT, 'logs', 'sweep_grf', 'grf_weight_scan')

# Top-3 energy combos from previous analysis
DEFAULT_COMBOS = [
    (0.04, 0.015, 0.0),
    (0.0, 0.0, 1e-5),
    (0.01, 0.03, 1e-6)
]

DEFAULT_WEIGHTS = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]


def run_one(bin_path, outdir, hind, front, internal, grf_w, run_idx):
    tag = f"h{hind:g}_f{front:g}_i{internal:g}_gw{grf_w:.0e}_r{run_idx}"
    csv_out = os.path.join(outdir, f"run_{tag}.csv")
    conv = csv_out.replace('.csv', '_conv.csv')
    env = os.environ.copy()
    env.update({
        'MJPC_CSV_LOG': csv_out,
        'MJPC_MAX_SIM_TIME': '45',
        'MJPC_GRF_HIND_WEIGHT': str(grf_w),
        'MJPC_GRF_FRONT_WEIGHT': str(grf_w),
        'MJPC_INTERNAL_GRF_ALIGN_WEIGHT': str(internal),
        'MJPC_CTRL_CLIP': env.get('MJPC_CTRL_CLIP', '10'),
        'MJPC_MIN_TRAVEL_DISTANCE_M': env.get('MJPC_MIN_TRAVEL_DISTANCE_M','0'),
    })
    print(f"Run -> hind={hind} front={front} internal={internal} grf_w={grf_w} run={run_idx}")
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

    return {'hind':hind,'front':front,'internal':internal,'grf_w':grf_w,'run':run_idx,'mp':mp,'mr':mr,'energy':en,'csv':conv if os.path.exists(conv) else csv_out,'status':status}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--combos', type=str, help='comma list of combos hind:front:internal e.g. 0.01:0.005:1e-4,...', default=None)
    p.add_argument('--weights', type=str, help='comma list of grf weights', default=None)
    p.add_argument('--run', action='store_true')
    p.add_argument('--bin', type=str, default=DEFAULT_BIN)
    p.add_argument('--outdir', type=str, default=OUTDIR)
    args = p.parse_args()

    if args.combos:
        combos = []
        for s in args.combos.split(','):
            h,f,i = s.split(':')
            combos.append((float(h), float(f), float(i)))
    else:
        combos = DEFAULT_COMBOS

    if args.weights:
        weights = [float(x) for x in args.weights.split(',')]
    else:
        weights = DEFAULT_WEIGHTS

    print('Combos to test (count={}):'.format(len(combos)))
    for c in combos:
        print(' ', c)
    print('GRF weights (count={}):'.format(len(weights)), weights)
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
        writer.writerow(['hind','front','internal','grf_w','run','mp','mr','energy','csv','status'])
        for (h,fr,i_w) in combos:
            for gw in weights:
                res = run_one(args.bin, args.outdir, h, fr, i_w, gw, 1)
                writer.writerow([res['hind'],res['front'],res['internal'],res['grf_w'],res['run'],res['mp'],res['mr'],res['energy'],res['csv'],res['status']])
                out_f.flush()
                rows.append(res)

    print('All runs completed. Wrote ->', raw_csv)


if __name__ == '__main__':
    main()
