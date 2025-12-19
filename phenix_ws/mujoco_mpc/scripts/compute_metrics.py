#!/usr/bin/env python3
"""Compute mean pitch/roll margins and energy for a converted dataset."""
import sys
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
    import csv
import importlib.util
from pathlib import Path

# load general_stability_margin
spec = importlib.util.spec_from_file_location(
    "general_stability_margin",
    str(Path(__file__).resolve().parent / "general_stability_margin.py"))
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)
get_value = gm.get_value
import os

# allow overriding analysis start/end window via environment variables
# (common workflow: analyze 10..60 s). Example:
# MJPC_METRICS_START_SEC=10 MJPC_METRICS_END_SEC=60 python3 compute_metrics.py ...
if os.environ.get('MJPC_METRICS_START_SEC') is not None:
    try:
        gm.start_time_sec = float(os.environ.get('MJPC_METRICS_START_SEC'))
    except Exception:
        pass
_METRICS_END_SEC = None
if os.environ.get('MJPC_METRICS_END_SEC') is not None:
    try:
        _METRICS_END_SEC = float(os.environ.get('MJPC_METRICS_END_SEC'))
    except Exception:
        _METRICS_END_SEC = None


def summarize(path):
    return summarize_with_min_travel(path, min_travel=5.0)


def summarize_with_min_travel(path, min_travel=0.5):
    # compute margins
    mp, mr = get_value(path)
    if mp.size == 0 or mr.size == 0:
        return None

    # load dataset and apply same trimming as general_stability_margin
    if pd is not None:
        ds = pd.read_csv(path, index_col=0, comment='#')
        try:
            ds.index = ds.index.astype(float)
        except Exception:
            pass
    else:
        # minimal CSV reader fallback
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # try to use the csv first column as numeric index
        # build a dict-of-lists that mimics pandas access
        ds = {}
        if rows:
            keys = rows[0].keys()
            for k in keys:
                ds[k] = [r[k] for r in rows]
            # index is the first column (time)
            try:
                ds_index = [float(r[next(iter(rows[0].keys()))]) for r in rows]
            except Exception:
                ds_index = None
    wall_crash_seconds = gm.wall_crash * gm.step
    if pd is not None:
        max_time = float(ds.index.max())
        # If an explicit end time was provided via env var, use it; otherwise
        # use the original wall_crash trimming behaviour.
        if _METRICS_END_SEC is not None:
            end_time = _METRICS_END_SEC
        else:
            end_time = max_time - wall_crash_seconds
        start_time = gm.start_time_sec
        ds_trim = ds[(ds.index >= start_time) & (ds.index <= end_time)]
        if ds_trim.shape[0] == 0:
            return None
    else:
        # ds is a dict-of-lists, use index values to slice
        if ds_index is None:
            return None
        max_time = max(ds_index)
        if _METRICS_END_SEC is not None:
            end_time = _METRICS_END_SEC
        else:
            end_time = max_time - wall_crash_seconds
        start_time = gm.start_time_sec
        mask = [(t >= start_time) and (t <= end_time) for t in ds_index]
        if not any(mask):
            return None
        def apply_mask(lst):
            return [v for v, keep in zip(lst, mask) if keep]
        ds_trim = {k: (apply_mask(v) if len(v) == len(mask) else v) for k, v in ds.items()}

    # check travel distance using COM (x,y)
    if pd is not None:
        body_xy_present = {'body_pose_position_x', 'body_pose_position_y'}.issubset(ds_trim.columns)
    else:
        body_xy_present = {'body_pose_position_x', 'body_pose_position_y'}.issubset(ds_trim.keys())

    if not body_xy_present:
        # can't compute travel, accept by default
        traveled = float('inf')
    else:
        try:
            if pd is not None:
                start = ds_trim[['body_pose_position_x', 'body_pose_position_y']].iloc[0].to_numpy()
                end = ds_trim[['body_pose_position_x', 'body_pose_position_y']].iloc[-1].to_numpy()
                traveled = float(np.linalg.norm(end - start))
            else:
                start_x = float(ds_trim['body_pose_position_x'][0])
                start_y = float(ds_trim['body_pose_position_y'][0])
                end_x = float(ds_trim['body_pose_position_x'][-1])
                end_y = float(ds_trim['body_pose_position_y'][-1])
                traveled = float(np.linalg.norm([end_x - start_x, end_y - start_y]))
        except Exception:
            traveled = float('nan')

    if traveled < min_travel:
        # mark as not valid because robot didn't travel enough
        raise RuntimeError(f'NO_TRAVEL (traveled={traveled:.3f} m < min={min_travel} m)')

    if pd is not None:
        energy = ds_trim['energy_abs_j'] if 'energy_abs_j' in ds_trim.columns else None
        energy_mean = float(energy.mean()) if energy is not None else None
    else:
        energy_mean = None
        if 'energy_abs_j' in ds_trim:
            try:
                vals = [float(v) for v in ds_trim['energy_abs_j'] if v != '']
                if vals:
                    energy_mean = float(np.mean(vals))
            except Exception:
                energy_mean = None

    return {
        'margin_pitch_mean': float(np.nanmean(mp)),
        'margin_roll_mean': float(np.nanmean(mr)),
        'energy_abs_mean': energy_mean,
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
