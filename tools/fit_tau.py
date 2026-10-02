"""Fit the building time constant tau from A/C-off decays of an indoor log (P1).

    python tools/fit_tau.py data/own_log.csv

Model: dT/dt = (T_out - T_in)/tau. Uses only A/C-off stretches whose indoor change is >= MIN_DECAY_C
(DHT11 is +/-2 C with integer resolution, so smaller decays are noise). Reports tau, residual RMS, n.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from logic.load_data import load_indoor_log  # noqa: E402

MIN_DECAY_C = 2.0


def off_segments(df):
    off = (df['ac_state'] == 0).to_numpy()
    segs, start = [], None
    for i, v in enumerate(off):
        if v and start is None:
            start = i
        if (not v or i == len(off) - 1) and start is not None:
            end = i if not v else i + 1
            segs.append((start, end))
            start = None
    return segs


def fit_tau(df, min_decay=MIN_DECAY_C):
    ts = ((df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600.0).to_numpy()   # hours
    tin, tout = df['indoor_temp_c'].to_numpy(float), df['outdoor_temp_c'].to_numpy(float)
    xs, ys, used = [], [], 0
    for a, b in off_segments(df):
        if b - a < 3 or abs(tin[b - 1] - tin[a]) < min_decay:
            continue
        used += 1
        dt = np.diff(ts[a:b])
        xs.append(((tout[a:b - 1] - tin[a:b - 1])))
        ys.append(np.diff(tin[a:b]) / dt)
    if not xs:
        return {'tau_h': None, 'segments_used': 0, 'note': f'no A/C-off decay >= {min_decay} C found'}
    x, y = np.concatenate(xs), np.concatenate(ys)
    k = float((x @ y) / (x @ x))           # 1/tau through the origin
    resid = y - k * x
    return {'tau_h': round(1.0 / k, 2) if k > 0 else None, 'segments_used': used,
            'residual_rms_c_per_h': round(float(np.sqrt(np.mean(resid ** 2))), 4), 'data_source': 'own_logger'}


if __name__ == '__main__':
    print(fit_tau(load_indoor_log(sys.argv[1])))
