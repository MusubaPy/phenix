#!/usr/bin/env python3
"""Simple gait evaluator for mjpc CSV runs.
Non-invasive: reads CSVs produced by mjpc_mod and computes summary metrics + optional plots.

Usage: scripts/gait_evaluator.py --csv path/to/run.csv [--log path/to/stdout.log] [--out path/to/out.json] [--plots path/to/plots_dir]
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def stdev(xs):
    if not xs:
        return float('nan')
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def read_csv(fn):
    with open(fn, 'r') as f:
        rdr = csv.DictReader(f)
        rows = [r for r in rdr]
    return rows, rdr.fieldnames


def to_float(s):
    try:
        return float(s)
    except Exception:
        return float('nan')


def compute_metrics(rows, fields):
    # collect columns
    data = defaultdict(list)
    for r in rows:
        for k in fields:
            data[k].append(r.get(k, ''))
    # numeric conversion convenience
    def colf(k):
        return [to_float(x) for x in data.get(k, [])]

    res = {}
    time = colf('time')
    res['duration'] = time[-1] - time[0] if len(time) >= 2 else 0.0

    com_x = colf('com_x')
    com_y = colf('com_y')
    com_z = colf('com_z')
    # distance: sum of segment lengths (xy plane)
    dist = 0.0
    for i in range(1, len(com_x)):
        dx = com_x[i] - com_x[i - 1]
        dy = com_y[i] - com_y[i - 1]
        dist += math.hypot(dx, dy)
    res['distance'] = dist

    energy = colf('energy_abs_j')
    res['mean_energy_abs'] = mean([x for x in energy if not math.isnan(x)])
    res['energy_per_m'] = res['mean_energy_abs'] / dist if dist > 0 else float('inf')

    res['mean_power_penalty'] = mean([x for x in colf('power_penalty_total') if not math.isnan(x)])
    res['mean_fixation_penalty'] = mean([x for x in colf('fixation_penalty_total') if not math.isnan(x)])

    # --- Robust energy computation: integrate applied torque * joint_vel over time ---
    # Find actuator columns (torque_applied_* and joint_vel_*). We pair them by
    # scanning fields in order to preserve actuator ordering.
    torque_cols = [c for c in fields if c.startswith('torque_applied_')]
    vel_cols = [c for c in fields if c.startswith('joint_vel_')]
    # Align lengths by occurrence order; if mismatch, take min length
    n_act = min(len(torque_cols), len(vel_cols))
    total_abs = 0.0
    total_signed = 0.0
    if len(time) >= 2 and n_act > 0:
        for i in range(1, len(time)):
            dt = time[i] - time[i - 1]
            if not math.isfinite(dt) or dt <= 0:
                continue
            step_abs = 0.0
            step_signed = 0.0
            for j in range(n_act):
                a = to_float(rows[i].get(torque_cols[j], '0'))
                v = to_float(rows[i].get(vel_cols[j], '0'))
                p = a * v
                step_signed += p * dt
                step_abs += abs(p) * dt
            total_abs += step_abs
            total_signed += step_signed
    res['total_energy_abs_j'] = total_abs
    res['total_energy_signed_j'] = total_signed
    res['energy_per_m_corrected'] = total_abs / dist if dist > 0 else float('inf')

    # Compare with logged cumulative column `energy_abs_j` if available
    energy_col = [to_float(r.get('energy_abs_j', 'nan')) for r in rows]
    if any(not math.isnan(x) for x in energy_col):
        # Use first and last finite entries
        finite = [x for x in energy_col if not math.isnan(x)]
        if finite:
            logged_delta = finite[-1] - finite[0]
            res['energy_abs_j_logged_delta'] = logged_delta
            # warn if logged cumulative differs substantially from integrated value
            if dist > 0 and abs((logged_delta - total_abs) / max(total_abs, 1e-8)) > 0.1:
                res['energy_consistency_warning'] = True
                res['energy_consistency_note'] = 'Logged cumulative energy differs from integrated actuator energy by >10%.'
            else:
                res['energy_consistency_warning'] = False
        else:
            res['energy_abs_j_logged_delta'] = None
            res['energy_consistency_warning'] = False
    else:
        res['energy_abs_j_logged_delta'] = None
        res['energy_consistency_warning'] = False

    # contact fractions per foot (using fz columns)
    feet = ['FL', 'HL', 'FR', 'HR']
    contact_frac = {}
    for ft in feet:
        col = f'grf_{ft}_fz'
        if col in fields:
            vals = colf(col)
            contact_frac[ft] = sum(1 for v in vals if v > 1.0) / len(vals) if vals else 0.0
        else:
            contact_frac[ft] = None
    res['contact_fraction'] = contact_frac

    # com_z stats
    res['com_z_mean'] = mean([x for x in com_z if not math.isnan(x)])
    res['com_z_std'] = stdev([x for x in com_z if not math.isnan(x)])

    # target switches (uses target_joint_* columns if available)
    target_joints = [c for c in fields if c.startswith('target_joint_')]
    t_changes = 0
    if target_joints:
        prev = None
        for i in range(len(rows)):
            cur_vals = tuple(rows[i].get(c, '') for c in target_joints)
            if i == 0:
                prev = cur_vals
                continue
            if cur_vals != prev:
                t_changes += 1
                prev = cur_vals
    res['target_switch_count'] = t_changes

    # target heights (if present)
    target_tz_cols = [c for c in fields if c.startswith('target_tz_')]
    target_tz_mean = {}
    for c in target_tz_cols:
        vals = [to_float(r.get(c, '')) for r in rows]
        target_tz_mean[c] = mean([v for v in vals if not math.isnan(v)])
    res['target_tz_mean'] = target_tz_mean

    # optional: if foot z explicit columns exist (foot_z_*), compute clearance
    footz_cols = [c for c in fields if c.startswith('foot_z_')]
    foot_clearance = {}
    for c in footz_cols:
        vals = [to_float(r.get(c, '')) for r in rows]
        # clearance defined as mean of [z when foot not in contact]
        foot_clearance[c] = mean([v for v in vals if not math.isnan(v)])
    res['foot_clearance_mean'] = foot_clearance

    return res


def parse_log(fn):
    # find NaN/Inf or "Rollout divergence" occurrences
    div_count = 0
    nan_count = 0
    if not fn or not os.path.exists(fn):
        return {'divergence_count': 0, 'nan_warnings': 0}
    with open(fn, 'r') as f:
        for line in f:
            if 'Rollout divergence' in line:
                div_count += 1
            if 'Nan, Inf' in line or 'NaN' in line or 'INF' in line or 'Inf' in line:
                nan_count += 1
    return {'divergence_count': div_count, 'nan_warnings': nan_count}


def save_json(outfn, data):
    d = os.path.dirname(outfn)
    if d and not os.path.exists(d):
        os.makedirs(d)
    with open(outfn, 'w') as f:
        json.dump(data, f, indent=2)


def plot_timeseries(rows, fields, outdir):
    if plt is None:
        print('matplotlib not available; skipping plots')
        return
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    time = [to_float(r.get('time', '')) for r in rows]
    # com_z
    if 'com_z' in fields:
        com_z = [to_float(r.get('com_z', '')) for r in rows]
        plt.figure()
        plt.plot(time, com_z)
        plt.xlabel('time')
        plt.ylabel('com_z')
        plt.title('COM z')
        plt.grid(True)
        plt.savefig(os.path.join(outdir, 'com_z.png'))
        plt.close()
    # energy
    if 'energy_abs_j' in fields:
        energy = [to_float(r.get('energy_abs_j', '')) for r in rows]
        plt.figure()
        plt.plot(time, energy)
        plt.xlabel('time')
        plt.ylabel('energy_abs_j')
        plt.title('Energy (abs)')
        plt.grid(True)
        plt.savefig(os.path.join(outdir, 'energy_abs_j.png'))
        plt.close()
    # grf z per foot
    for ft in ['FL', 'HL', 'FR', 'HR']:
        col = f'grf_{ft}_fz'
        if col in fields:
            vals = [to_float(r.get(col, '')) for r in rows]
            plt.figure()
            plt.plot(time, vals)
            plt.xlabel('time')
            plt.ylabel(col)
            plt.title(col)
            plt.grid(True)
            plt.savefig(os.path.join(outdir, f'{col}.png'))
            plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Path to CSV file')
    ap.add_argument('--log', required=False, help='Optional log/stdout file to parse for divergence warnings')
    ap.add_argument('--out', required=False, default=None, help='Output json filename (default: CSVname_eval.json)')
    ap.add_argument('--plots', required=False, default=None, help='Output directory for plots (optional)')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print('CSV not found:', args.csv, file=sys.stderr)
        sys.exit(2)

    rows, fields = read_csv(args.csv)
    metrics = compute_metrics(rows, fields)
    log_metrics = parse_log(args.log) if args.log else {'divergence_count': 0, 'nan_warnings': 0}
    report = {
        'csv': args.csv,
        'rows': len(rows),
        'fields': fields,
        'metrics': metrics,
        'log': log_metrics,
    }

    outfn = args.out or (os.path.splitext(args.csv)[0] + '_eval.json')
    save_json(outfn, report)
    print('Wrote report to', outfn)

    if args.plots:
        plot_timeseries(rows, fields, args.plots)
        print('Saved plots to', args.plots)


if __name__ == '__main__':
    main()
