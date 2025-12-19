#!/usr/bin/env python3
"""Convert mjpc CSV output to a dataset compatible with
`general_stability_margin.get_value` expected column names.
"""
import sys
try:
    import pandas as pd
except Exception:
    pd = None
    import csv
    from collections import OrderedDict

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

    if pd is not None:
        df = pd.read_csv(inp, index_col=0, comment='#')
    else:
        # minimal csv reader fallback (no pandas)
        with open(inp, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            cols = reader.fieldnames
        # build a simple dict-of-lists similar to pandas
        df = OrderedDict()
        for c in cols:
            df[c] = [row.get(c, '') for row in rows]

    # create new df/dict with required columns
    if pd is not None:
        new = pd.DataFrame(index=df.index)
    else:
        new = OrderedDict()

    # map body/com position
    def has_col(name):
        if pd is not None:
            return name in df.columns
        return name in df

    def get_col(name):
        if pd is not None:
            return df[name]
        return df[name]

    if has_col('com_x') and has_col('com_y') and has_col('com_z'):
        new['body_pose_position_x'] = get_col('com_x')
        new['body_pose_position_y'] = get_col('com_y')
        new['body_pose_position_z'] = get_col('com_z')
    else:
        raise RuntimeError('Expected com_x/com_y/com_z columns in mjpc CSV')

    # compute distance traveled from start (2D horizontal displacement)
    try:
        if pd is not None:
            start_xy = df[['com_x', 'com_y']].iloc[0].to_numpy()
            displacement = (df[['com_x', 'com_y']].to_numpy() - start_xy)
            dist = (displacement ** 2).sum(axis=1) ** 0.5
            new['distance_traveled_m'] = dist
        else:
            cx = [float(v) if v != '' else float('nan') for v in df['com_x']]
            cy = [float(v) if v != '' else float('nan') for v in df['com_y']]
            sx, sy = cx[0], cy[0]
            dist = [((x - sx)**2 + (y - sy)**2)**0.5 for x, y in zip(cx, cy)]
            new['distance_traveled_m'] = dist
    except Exception:
        new['distance_traveled_m'] = float('nan')

    # map GRF columns
    for leg_idx in range(4):
        src = MAP[f'grf_leg{leg_idx}']
        # expected columns like grf_FL_fx
        for dim, col in zip(['x','y','z'], ['_fx','_fy','_fz']):
            src_col = f'{src}{col}'
            dst_col = f'grf_leg{leg_idx}_{dim}'
            if has_col(src_col):
                new[dst_col] = get_col(src_col)
            else:
                # if missing, fill with zeros
                if pd is not None:
                    import numpy as _np
                    new[dst_col] = _np.zeros(len(df))
                else:
                    new[dst_col] = [0.0] * len(next(iter(df.values())))

    # copy energy metrics if present
    if has_col('energy_abs_j'):
        new['energy_abs_j'] = get_col('energy_abs_j')
    if has_col('energy_signed_j'):
        new['energy_signed_j'] = get_col('energy_signed_j')

    if pd is not None:
        new.to_csv(out)
    else:
        # write via csv.DictWriter (ensures a single clean header line)
        keys = list(new.keys())
        # normalize keys to ensure no accidental whitespace/newlines
        keys = [k.strip() for k in keys]
        # determine number of rows from any column
        nrows = len(next(iter(new.values()))) if len(new) else 0
        with open(out, 'w', newline='') as f:
            # write a single-line header explicitly and ensure newline-only
            # behaviour (no CRLF). Use csv.writer with lineterminator='\n'
            # so rows also use LF endings.
            f.write(','.join(keys) + '\n')
            w = csv.writer(f, lineterminator='\n')
            for i in range(nrows):
                row = [new[k][i] for k in keys]
                w.writerow(row)
    print(f'Converted {inp} -> {out}')
