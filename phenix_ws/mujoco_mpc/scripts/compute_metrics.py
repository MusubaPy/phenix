#!/usr/bin/env python3
"""Compute mean pitch/roll margins and energy for a converted dataset."""
import sys
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path

# load general_stability_margin
spec = importlib.util.spec_from_file_location(
    "general_stability_margin",
    str(Path(__file__).resolve().parent / "general_stability_margin.py"))
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)
get_value = gm.get_value


def summarize(path):
    return summarize_with_min_travel(path, min_travel=5.0)


def summarize_with_min_travel(path, min_travel=0.5):
    # compute margins
    mp, mr = get_value(path)
    if mp.size == 0 or mr.size == 0:
        return None

    # load dataset and apply same trimming as general_stability_margin
    ds = pd.read_csv(path, index_col=0, comment='#')
    try:
        ds.index = ds.index.astype(float)
    except Exception:
        pass
    wall_crash_seconds = gm.wall_crash * gm.step
    max_time = float(ds.index.max())
    end_time = max_time - wall_crash_seconds
    ds_trim = ds[(ds.index >= gm.start_time_sec) & (ds.index <= end_time)]
    if ds_trim.shape[0] == 0:
        return None

    # check travel distance using COM (x,y)
    if not {'body_pose_position_x', 'body_pose_position_y'}.issubset(ds_trim.columns):
        # can't compute travel, accept by default
        traveled = float('inf')
    else:
        start = ds_trim[['body_pose_position_x', 'body_pose_position_y']].iloc[0].to_numpy()
        end = ds_trim[['body_pose_position_x', 'body_pose_position_y']].iloc[-1].to_numpy()
        traveled = float(np.linalg.norm(end - start))

    if traveled < min_travel:
        # mark as not valid because robot didn't travel enough
        raise RuntimeError(f'NO_TRAVEL (traveled={traveled:.3f} m < min={min_travel} m)')

    energy = ds_trim['energy_abs_j'] if 'energy_abs_j' in ds_trim.columns else None
    return {
        'margin_pitch_mean': float(np.nanmean(mp)),
        'margin_roll_mean': float(np.nanmean(mr)),
        'energy_abs_mean': float(energy.mean()) if energy is not None else None,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: compute_metrics.py <converted_dataset.csv>')
        sys.exit(1)
    path = sys.argv[1]
    # optional minimum travel distance (meters). Can be set via env var or CLI
    import os
    # default minimum travel distance is 5 meters (user preference)
    min_travel = float(os.environ.get('MJPC_MIN_TRAVEL_DISTANCE_M', '5.0'))
    try:
        s = summarize_with_min_travel(path, min_travel=min_travel)
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith('NO_TRAVEL'):
            print(msg)
            sys.exit(3)
        else:
            raise
    if s is None:
        print('NO_DATA')
        sys.exit(2)
    print(f"{s['margin_pitch_mean']:.6f},{s['margin_roll_mean']:.6f},{s['energy_abs_mean']:.6f}")
