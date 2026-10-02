import numpy as np
import pandas as pd
import pytest

import config
from logic import forecast
from logic.load_data import load_data, load_indoor_log, _require_hourly_cols
from tools.fit_tau import fit_tau, write_override


def test_phase_shift_applied_and_logged(df):
    assert df.attrs['provenance']['phase_shift_h'] == config.OUTDOOR_PHASE_SHIFT_H
    assert df.groupby('hour')['temp'].mean().idxmax() in (13, 14, 15)
    assert df[['temp', 'ac_kwh']].corr().iloc[0, 1] > 0.7
    assert (df['data_source'] == 'synthetic_sample').all()


def test_required_outdoor_cols_present(df):
    for c in config.REQUIRED_HOURLY_COLS:
        assert c in df.columns
    assert 'indoor_temp_c' not in df.columns and 'indoor_temp_est_c' not in df.columns


def test_forecast_features_are_outdoor_only():
    assert 'temp' in forecast.FEATURES
    assert 'ac_lag_24h' in forecast.FEATURES
    low = [f.lower() for f in forecast.FEATURES]
    for bad in forecast.FORBIDDEN_FEATURE_SUBSTR:
        assert not any(bad in f for f in low)


def test_load_data_rejects_missing_outdoor(tmp_path):
    p = tmp_path / 'bad.csv'
    pd.DataFrame({'timestamp': pd.date_range('2025-07-01', periods=24, freq='h'),
                  'ac_kwh': 1.0}).to_csv(p, index=False)
    raw = pd.read_csv(p)
    raw.columns = [c.lower() for c in raw.columns]
    with pytest.raises(ValueError, match='missing required columns'):
        _require_hourly_cols(raw, p)


def test_fit_tau_recovers_known_value(tmp_path):
    tau, n = 6.0, 400
    ts = pd.date_range('2026-01-01', periods=n, freq='5min')
    tout = np.full(n, 40.0)
    tin = np.empty(n); tin[0] = 24.0
    for i in range(1, n):
        tin[i] = tin[i - 1] + (tout[i - 1] - tin[i - 1]) / tau * (5 / 60)
    p = tmp_path / 'log.csv'
    pd.DataFrame({'timestamp': ts, 'indoor_temp_c': tin, 'indoor_rh_pct': 50, 'outdoor_temp_c': tout, 'ac_state': 0}).to_csv(p, index=False)
    r = fit_tau(load_indoor_log(p))
    assert abs(r['tau_h'] - tau) < 0.3 and r['data_source'] == 'own_logger'
    out = tmp_path / 'override.json'
    write_override(r, path=str(out))
    assert out.is_file()
