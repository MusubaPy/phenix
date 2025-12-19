#!/usr/bin/env python3
"""Compute mechanical energy and heat losses from mjpc CSV logs.

Usage:
  scripts/compute_mech_heat.py --csv file1.csv [file2.csv ...] [--tmin 10.0] [--kt 0.9287 --ra 0.4 --gs 1.0] [--out out.json]

Prints a summary per file and optionally writes JSON with detailed numbers.
"""

import argparse
import csv
import json
import math
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', nargs='+', required=True, help='One or more MJPC CSV logs')
    p.add_argument('--tmin', type=float, default=10.0, help='Minimum time to include (seconds)')
    p.add_argument('--kt', type=float, default=0.9287, help='Motor torque constant Kt')
    p.add_argument('--ra', type=float, default=0.4, help='Motor armature resistance Ra')
    p.add_argument('--gs', type=float, default=1.0, help='Gear ratio Gs')
    p.add_argument('--out', type=str, default=None, help='Optional JSON output file')
    return p.parse_args()


def load_header_and_rows(fn):
    with open(fn, 'r') as f:
        header = None
        # skip initial comment lines
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith('#'):
                continue
            header = line.strip()
            # Reset reader so csv.DictReader sees the header as first line
            f.seek(pos)
            break
        if header is None:
            raise RuntimeError('No header found in %s' % fn)
        rdr = csv.DictReader(f)
        rows = [r for r in rdr]
    return rdr.fieldnames, rows


def compute_mech_heat(rows, fields, tmin, heat_coef):
    # find columns
    torque_cols = [c for c in fields if c.startswith('torque_applied_')]
    vel_cols = [c for c in fields if c.startswith('joint_vel_')]
    n_act = min(len(torque_cols), len(vel_cols))

    # collect rows with time >= tmin
    times = []
    rows_f = []
    for r in rows:
        try:
            t = float(r.get('time', 'nan'))
        except Exception:
            continue
        if not math.isfinite(t):
            continue
        if t < tmin:
            continue
        times.append(t)
        rows_f.append(r)
    if len(times) < 2:
        return None

    mech = 0.0
    heat = 0.0
    signed = 0.0

    # integrate as in gait_evaluator: use current row values for power
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if not math.isfinite(dt) or dt <= 0:
            continue
        step_abs = 0.0
        step_signed = 0.0
        step_heat = 0.0
        for j in range(n_act):
            try:
                a = float(rows_f[i].get(torque_cols[j], 'nan'))
                v = float(rows_f[i].get(vel_cols[j], 'nan'))
            except Exception:
                a = float('nan'); v = float('nan')
            if math.isnan(a) or math.isnan(v):
                continue
            p = a * v
            step_signed += p * dt
            step_abs += abs(p) * dt
            step_heat += (a * a) * dt * heat_coef
        mech += step_abs
        signed += step_signed
        heat += step_heat

    # distance for normalization
    comx = [float(r.get('com_x', 'nan')) for r in rows_f]
    comy = [float(r.get('com_y', 'nan')) for r in rows_f]
    dist = 0.0
    for i in range(1, len(comx)):
        if math.isnan(comx[i]) or math.isnan(comx[i-1]):
            continue
        dx = comx[i] - comx[i - 1]
        dy = comy[i] - comy[i - 1]
        dist += math.hypot(dx, dy)

    return {
        'mechanical_J': mech,
        'heat_J': heat,
        'total_J': mech + heat,
        'signed_J': signed,
        'distance_m': dist,
        'mechanical_per_m': mech / dist if dist > 0 else float('inf'),
        'total_per_m': (mech + heat) / dist if dist > 0 else float('inf'),
    }


def main():
    args = parse_args()
    heat_coef = args.ra * (args.gs ** 2) / (args.kt ** 2)

    out = {}
    for fn in args.csv:
        if not os.path.exists(fn):
            print('File not found:', fn, file=sys.stderr)
            continue
        try:
            fields, rows = load_header_and_rows(fn)
            res = compute_mech_heat(rows, fields, args.tmin, heat_coef)
            if res is None:
                print('Not enough data >= tmin for', fn)
                continue
            print('\n---- {} ----'.format(fn))
            print('Mechanical energy (∫ |τ·θ̇| dt) = {:.6f} J'.format(res['mechanical_J']))
            print('Heat losses (coef={:.6e}) = {:.6f} J'.format(heat_coef, res['heat_J']))
            print('Total (mech + heat) = {:.6f} J'.format(res['total_J']))
            print('Signed energy (∫ τ·θ̇ dt) = {:.6f} J'.format(res['signed_J']))
            print('Distance traveled = {:.6f} m'.format(res['distance_m']))
            if res['distance_m'] > 0:
                print('Mechanical per meter = {:.6f} J/m'.format(res['mechanical_per_m']))
                print('Total per meter = {:.6f} J/m'.format(res['total_per_m']))
            out[fn] = res
        except Exception as e:
            print('Error processing', fn, ':', e, file=sys.stderr)

    if args.out:
        d = os.path.dirname(args.out)
        if d and not os.path.exists(d):
            os.makedirs(d)
        with open(args.out, 'w') as f:
            json.dump({'heat_coef': heat_coef, 'results': out}, f, indent=2)
        print('\nWrote results to', args.out)


if __name__ == '__main__':
    main()
