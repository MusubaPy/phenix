# Adapted general stability margin tool
# (content based on the user's attached script - trimmed to essential functions)
try:
    import pandas as pd
except Exception:
    pd = None
    import csv
import numpy as np

# Constants used in the original script
step = 2e-3
smoothing_window = 3
sampling_rate = int(1/step)
# start analysis after this many seconds to allow gait to normalize
start_time_sec = 10.0
# trailing trim in samples (original script used 500 samples)
wall_crash = 500

body_length = 0.5
body_width = 0.32


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm


def get_value(dataset_name: str):
    if pd is not None:
        ds = pd.read_csv(dataset_name, index_col=0, comment='#')
        # ensure time index is numeric
        try:
            ds.index = ds.index.astype(float)
        except Exception:
            pass

        # trim end: convert wall_crash samples -> seconds
        wall_crash_seconds = wall_crash * step
        max_time = float(ds.index.max())
        end_time = max_time - wall_crash_seconds

        # filter by time: start at `start_time_sec`, end at `end_time`
        ds = ds[(ds.index >= start_time_sec) & (ds.index <= end_time)]

        # if dataset is empty after trimming, return empty arrays
        if ds.shape[0] == 0:
            return np.array([]), np.array([])

        grf_leg0_x = ds['grf_leg0_x'].tolist()
        grf_leg1_x = ds['grf_leg1_x'].tolist()
        grf_leg2_x = ds['grf_leg2_x'].tolist()
        grf_leg3_x = ds['grf_leg3_x'].tolist()

        grf_leg0_y = ds['grf_leg0_y'].tolist()
        grf_leg1_y = ds['grf_leg1_y'].tolist()
        grf_leg2_y = ds['grf_leg2_y'].tolist()
        grf_leg3_y = ds['grf_leg3_y'].tolist()

        grf_leg0_z = ds['grf_leg0_z'].tolist()
        grf_leg1_z = ds['grf_leg1_z'].tolist()
        grf_leg2_z = ds['grf_leg2_z'].tolist()
        grf_leg3_z = ds['grf_leg3_z'].tolist()

        f_net_x = [grf_leg0_x[i] + grf_leg1_x[i] + grf_leg2_x[i] + grf_leg3_x[i] for i in range(0, len(grf_leg1_x))]
        f_net_y = [grf_leg0_y[i] + grf_leg1_y[i] + grf_leg2_y[i] + grf_leg3_y[i] for i in range(0, len(grf_leg1_y))]
        f_net_z = [grf_leg0_z[i] + grf_leg1_z[i] + grf_leg2_z[i] + grf_leg3_z[i] for i in range(0, len(grf_leg1_z))]

        body_position_x = ds['body_pose_position_x'].tolist()
        body_position_y = ds['body_pose_position_y'].tolist()
        body_position_z = ds['body_pose_position_z'].tolist()
    else:
        # minimal csv reader fallback
        with open(dataset_name, newline='') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]
        # try to use time index and filter by time
        if not rows:
            return np.array([]), np.array([])
        times = []
        for r in rows:
            try:
                times.append(float(next(iter(r.values()))))
            except Exception:
                times.append(float('nan'))

        wall_crash_seconds = wall_crash * step
        max_time = max(times)
        end_time = max_time - wall_crash_seconds
        # filter rows
        rows_trim = [r for t, r in zip(times, rows) if (t >= start_time_sec and t <= end_time)]
        if not rows_trim:
            return np.array([]), np.array([])

        def get_list(col):
            return [float(r[col]) if r.get(col, '') != '' else 0.0 for r in rows_trim]

        try:
            grf_leg0_x = get_list('grf_leg0_x')
            grf_leg1_x = get_list('grf_leg1_x')
            grf_leg2_x = get_list('grf_leg2_x')
            grf_leg3_x = get_list('grf_leg3_x')

            grf_leg0_y = get_list('grf_leg0_y')
            grf_leg1_y = get_list('grf_leg1_y')
            grf_leg2_y = get_list('grf_leg2_y')
            grf_leg3_y = get_list('grf_leg3_y')

            grf_leg0_z = get_list('grf_leg0_z')
            grf_leg1_z = get_list('grf_leg1_z')
            grf_leg2_z = get_list('grf_leg2_z')
            grf_leg3_z = get_list('grf_leg3_z')

            f_net_x = [grf_leg0_x[i] + grf_leg1_x[i] + grf_leg2_x[i] + grf_leg3_x[i] for i in range(0, len(grf_leg1_x))]
            f_net_y = [grf_leg0_y[i] + grf_leg1_y[i] + grf_leg2_y[i] + grf_leg3_y[i] for i in range(0, len(grf_leg1_y))]
            f_net_z = [grf_leg0_z[i] + grf_leg1_z[i] + grf_leg2_z[i] + grf_leg3_z[i] for i in range(0, len(grf_leg1_z))]

            body_position_x = get_list('body_pose_position_x')
            body_position_y = get_list('body_pose_position_y')
            body_position_z = get_list('body_pose_position_z')
        except Exception:
            return np.array([]), np.array([])

    contacts = np.where(grf_leg0_z > np.mean(grf_leg0_z), 1, 0)

    margin_roll = np.zeros_like(f_net_x)
    margin_pitch = np.zeros_like(f_net_x)

    for i in range(0, len(margin_pitch)):
        CoM = np.array([body_position_x[i], body_position_y[i], body_position_z[i]])
        f_net = np.array([f_net_x[i], f_net_y[i], f_net_z[i]])

        FL = np.array([body_position_x[i] + body_length/2, body_position_y[i] + body_width/2, body_position_z[i] - 0.275])
        BL = np.array([body_position_x[i] - body_length/2, body_position_y[i] + body_width/2, body_position_z[i] - 0.275])
        FR = np.array([body_position_x[i] + body_length/2, body_position_y[i] - body_width/2, body_position_z[i] - 0.275])
        BR = np.array([body_position_x[i] - body_length/2, body_position_y[i] - body_width/2, body_position_z[i] - 0.275])

        # Tipover axes
        a_front = FR - FL
        a_right = BR - FR
        a_back = BL - BR
        a_left = FL - BL

        # Normalizing
        a_front_hat = normalize(a_front)
        a_right_hat = normalize(a_right)
        a_back_hat = normalize(a_back)
        a_left_hat = normalize(a_left)

        l_front = (np.eye(3) - np.outer(a_front_hat, a_front_hat.T)) @ (FR - CoM)
        l_right = (np.eye(3) - np.outer(a_right_hat, a_right_hat.T)) @ (BR - CoM)
        l_back = (np.eye(3) - np.outer(a_back_hat, a_back_hat.T)) @ (BL - CoM)
        l_left = (np.eye(3) - np.outer(a_left_hat, a_left_hat.T)) @ (FL - CoM)

        l_front_hat = normalize(l_front)
        l_right_hat = normalize(l_right)
        l_back_hat = normalize(l_back)
        l_left_hat = normalize(l_left)

        if contacts[i] < 1:
            support_line_center = np.array([(FR[0] + BL[0])/2, (FR[1] + BL[1])/2, body_position_z[i] - 0.275])
        else:
            support_line_center = np.array([(FL[0] + BR[0])/2, (FL[1] + BR[1])/2, body_position_z[i] - 0.275])

        f_r = support_line_center - f_net
        r = CoM - support_line_center
        n_r = np.cross(r, f_r)

        f_front = (np.eye(3) - np.outer(a_front_hat, a_front_hat.T)) @ f_r
        f_right = (np.eye(3) - np.outer(a_right_hat, a_right_hat.T)) @ f_r
        f_back = (np.eye(3) - np.outer(a_back_hat, a_back_hat.T)) @ f_r
        f_left = (np.eye(3) - np.outer(a_left_hat, a_left_hat.T)) @ f_r

        n_front = np.outer(a_front_hat, a_front_hat.T) @ n_r
        n_right = np.outer(a_right_hat, a_right_hat.T) @ n_r
        n_back = np.outer(a_back_hat, a_back_hat.T) @ n_r
        n_left = np.outer(a_left_hat, a_left_hat.T) @ n_r

        f_front_star = f_front + np.cross(l_front_hat, n_front)/np.linalg.norm(l_front_hat)
        f_right_star = f_right + np.cross(l_right_hat, n_right)/np.linalg.norm(l_right_hat)
        f_back_star = f_back + np.cross(l_back_hat, n_back)/np.linalg.norm(l_back_hat)
        f_left_star = f_left + np.cross(l_left_hat, n_left)/np.linalg.norm(l_left_hat)

        f_front_star_hat = normalize(f_front_star)
        f_right_star_hat = normalize(f_right_star)
        f_back_star_hat = normalize(f_back_star)
        f_left_star_hat = normalize(f_left_star)

        # compute angles
        def angle(a, b):
            a_dot_b = np.dot(a, b)
            a_dot_b = np.clip(a_dot_b, -1.0, 1.0)
            return np.arccos(a_dot_b)

        alpha_front = angle(f_front_star_hat, l_front_hat)
        alpha_right = angle(f_right_star_hat, l_right_hat)
        alpha_back = angle(f_back_star_hat, l_back_hat)
        alpha_left = angle(f_left_star_hat, l_left_hat)

        margin_pitch[i] = min(alpha_front, alpha_back)
        margin_roll[i] = min(alpha_left, alpha_right)

    return margin_pitch, margin_roll
