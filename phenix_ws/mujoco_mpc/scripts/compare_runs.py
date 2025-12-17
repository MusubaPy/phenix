#!/usr/bin/env python3
"""Compare two mjpc runs (with/without our cost) using margins and energy.

Usage: compare_runs.py <converted_run_with.csv> <converted_run_without.csv>
"""
import sys
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path

# load local general_stability_margin from scripts/
spec = importlib.util.spec_from_file_location(
    "general_stability_margin",
    str(Path(__file__).resolve().parent / "general_stability_margin.py"))
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)
get_value = gm.get_value


def summarize(path):
    mp, mr = get_value(path)
    if mp.size == 0 or mr.size == 0:
        return None
    df = pd.read_csv(path, index_col=0)
    energy = df['energy_abs_j'] if 'energy_abs_j' in df.columns else None
    return {
        'margin_pitch_mean': float(np.nanmean(mp)),
        'margin_roll_mean': float(np.nanmean(mr)),
        'energy_abs_mean': float(energy.mean()) if energy is not None else None,
    }


def pretty(d):
    return (f"pitch={d['margin_pitch_mean']:.4f}, roll={d['margin_roll_mean']:.4f}, "
            f"energy_abs={d['energy_abs_mean']:.3f}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: compare_runs.py <with.csv> <without.csv>')
        sys.exit(1)

    a = sys.argv[1]
    b = sys.argv[2]

    sa = summarize(a)
    sb = summarize(b)

    if sa is None:
        print(f'No valid data after {gm.start_time_sec}s in {a}.')
    if sb is None:
        print(f'No valid data after {gm.start_time_sec}s in {b}.')
    if sa is None or sb is None:
        sys.exit(1)

    print('With cost:   ', pretty(sa))
    print('Without cost:', pretty(sb))

    print('\nDifferences (with - without):')
    print(f"pitch: {sa['margin_pitch_mean'] - sb['margin_pitch_mean']:.4f}")
    print(f"roll:  {sa['margin_roll_mean'] - sb['margin_roll_mean']:.4f}")
    if sa['energy_abs_mean'] is not None and sb['energy_abs_mean'] is not None:
        print(f"energy_abs: {sa['energy_abs_mean'] - sb['energy_abs_mean']:.3f}")
