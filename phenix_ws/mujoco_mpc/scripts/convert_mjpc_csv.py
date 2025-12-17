#!/usr/bin/env python3
"""Convert mjpc CSV output to a dataset compatible with
`general_stability_margin.get_value` expected column names.
"""
import sys
import pandas as pd

MAP = {
    'grf_leg0': 'grf_FL',
    'grf_leg1': 'grf_HL',
    'grf_leg2': 'grf_FR',
    'grf_leg3': 'grf_HR',
}

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: convert_mjpc_csv.py <input_mjpc_csv> <output_dataset_csv>')
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2]

    df = pd.read_csv(inp, index_col=0, comment='#')

    # create new df with required columns
    new = pd.DataFrame(index=df.index)

    # map body/com position
    if 'com_x' in df.columns and 'com_y' in df.columns and 'com_z' in df.columns:
        new['body_pose_position_x'] = df['com_x']
        new['body_pose_position_y'] = df['com_y']
        new['body_pose_position_z'] = df['com_z']
    else:
        raise RuntimeError('Expected com_x/com_y/com_z columns in mjpc CSV')

    # compute distance traveled from start (2D horizontal displacement)
    try:
        start_xy = df[['com_x', 'com_y']].iloc[0].to_numpy()
        displacement = (df[['com_x', 'com_y']].to_numpy() - start_xy)
        dist = (displacement ** 2).sum(axis=1) ** 0.5
        new['distance_traveled_m'] = dist
    except Exception:
        # if anything fails, fill with NaN
        new['distance_traveled_m'] = float('nan')

    # map GRF columns
    for leg_idx in range(4):
        src = MAP[f'grf_leg{leg_idx}']
        # expected columns like grf_FL_fx
        for dim, col in zip(['x','y','z'], ['_fx','_fy','_fz']):
            src_col = f'{src}{col}'
            dst_col = f'grf_leg{leg_idx}_{dim}'
            if src_col in df.columns:
                new[dst_col] = df[src_col]
            else:
                # if missing, fill with zeros
                new[dst_col] = 0.0

    # copy energy metrics if present
    if 'energy_abs_j' in df.columns:
        new['energy_abs_j'] = df['energy_abs_j']
    if 'energy_signed_j' in df.columns:
        new['energy_signed_j'] = df['energy_signed_j']

    new.to_csv(out)
    print(f'Converted {inp} -> {out}')
