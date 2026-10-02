"""Day-ahead A/C load forecaster (7.6). No lag shorter than 24 h. Weather forecast assumed equal to actual."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import config

FEATURES = ['hour', 'day_of_week', 'is_weekend', 'temp', 'humidity', 'dew_point', 'solar',
            'ac_lag_24h', 'prev_day_mean', 'prev_day_max']


def build_forecast_features(df):
    """Adds day-ahead features. Day 1 has no 24 h history and is dropped by callers via `has_history`."""
    df = df.sort_values('timestamp').reset_index(drop=True).copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([4, 5]).astype(int)
    df['day'] = df['timestamp'].dt.date.astype(str)
    df['ac_lag_24h'] = df['ac_kwh'].shift(24)
    g = df.groupby('day')['ac_kwh']
    daily = pd.DataFrame({'m': g.mean(), 'x': g.max()}).shift(1)
    df['prev_day_mean'] = df['day'].map(daily['m'])
    df['prev_day_max'] = df['day'].map(daily['x'])
    df['has_history'] = df[['ac_lag_24h', 'prev_day_mean', 'prev_day_max']].notna().all(axis=1)
    return df


def split_days(df):
    days = sorted(df.loc[df['has_history'], 'day'].unique())
    test = days[-config.TEST_DAYS:]
    train = [d for d in days if d not in test]
    return train, test


def train_forecaster(df_feat, train_days=None):
    if train_days is None:
        train_days, _ = split_days(df_feat)
    tr = df_feat[df_feat['day'].isin(train_days)]
    model = RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_SEED, min_samples_leaf=2)
    model.fit(tr[FEATURES], tr['ac_kwh'])
    return model


def forecast_frame(df):
    """Returns df (days with history only) with forecast columns and a split label per row."""
    df_feat = build_forecast_features(df)
    train_days, test_days = split_days(df_feat)
    model = train_forecaster(df_feat, train_days)
    d = df_feat[df_feat['has_history']].copy()
    d['predicted_kwh'] = np.clip(model.predict(d[FEATURES]), 0.0, config.AC_MAX_KWH).round(4)
    d['naive_yesterday_kwh'] = d['ac_lag_24h']
    hod = df_feat[df_feat['day'].isin(train_days)].groupby('hour')['ac_kwh'].mean()
    d['naive_hod_mean_kwh'] = d['hour'].map(hod)
    d['forecast_split'] = np.where(d['day'].isin(test_days), 'test', 'train_in_sample')
    return d, model, {'train_days': train_days, 'test_days': test_days}


def forecast_metrics(d):
    t = d[d['forecast_split'] == 'test']
    out = {'n_test_hours': int(len(t)), 'test_days': int(t['day'].nunique())}
    for name, col in [('rf', 'predicted_kwh'), ('naive_same_hour_yesterday', 'naive_yesterday_kwh'),
                      ('naive_hour_of_day_mean', 'naive_hod_mean_kwh')]:
        out[name] = {'mae_kwh': round(float(mean_absolute_error(t['ac_kwh'], t[col])), 4),
                     'r2': round(float(r2_score(t['ac_kwh'], t[col])), 4)}
    best_naive = min(out['naive_same_hour_yesterday']['mae_kwh'], out['naive_hour_of_day_mean']['mae_kwh'])
    out['rf_beats_best_naive_mae'] = bool(out['rf']['mae_kwh'] < best_naive)
    return out


def detect_peak_window(forecast_24):
    """Contiguous hours >= DETECT_FRACTION of the forecast daily max, around the max hour."""
    f = np.asarray(forecast_24, dtype=float)
    thr = config.DETECT_FRACTION * f.max()
    i = int(f.argmax())
    lo = hi = i
    while lo > 0 and f[lo - 1] >= thr:
        lo -= 1
    while hi < 23 and f[hi + 1] >= thr:
        hi += 1
    return list(range(lo, hi + 1))


def window_overlap(detected, tariff=config.PEAK_HOURS):
    a, b = set(detected), set(tariff)
    return round(len(a & b) / len(a | b), 3) if (a | b) else 0.0
