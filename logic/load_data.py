import glob
import os

import pandas as pd

import config

_RENAME = {
    'temperature_c': 'temp',
    'humidity_pct': 'humidity',
    'dew_point_c': 'dew_point',
    'ac_consumption_kwh': 'ac_kwh',
    'total_consumption_kwh': 'total_kwh',
    'solar_irradiance_wm2': 'solar',
    'is_peak_hour': 'is_peak',
}
# organizer-file synonyms (small schema adapter, P2: untested on a real file)
_ORGANIZER_SYNONYMS = {
    'temperature': 'temp', 'temp_c': 'temp', 'outdoor_temp': 'temp',
    'humidity_percent': 'humidity', 'rh': 'humidity', 'dewpoint': 'dew_point', 'dew_point_celsius': 'dew_point',
    'ac_kwh': 'ac_kwh', 'ac_consumption': 'ac_kwh', 'ac_energy_kwh': 'ac_kwh',
    'total_kwh': 'total_kwh', 'total_consumption': 'total_kwh',
    'solar_irradiance': 'solar', 'solar_wm2': 'solar', 'ghi': 'solar',
    'datetime': 'timestamp', 'time': 'timestamp', 'house_id': 'building_id', 'home_id': 'building_id',
}


def _rotate_within_day(df, cols, shift_h):
    """Value at hour h becomes the original value at hour (h + shift) % 24, per calendar day."""
    if shift_h % 24 == 0:
        return df
    df = df.sort_values('timestamp').reset_index(drop=True)
    day = df['timestamp'].dt.normalize()
    for c in cols:
        out = df[c].copy()
        for _, idx in df.groupby(day).groups.items():
            idx = list(idx)
            if len(idx) == 24:
                vals = df.loc[idx, c].to_numpy()
                hrs = df.loc[idx, 'timestamp'].dt.hour.to_numpy()
                src = {h: v for h, v in zip(hrs, vals)}
                out.loc[idx] = [src[(h + shift_h) % 24] for h in hrs]
        df[c] = out
    return df


def _find_organizer_file():
    files = sorted(glob.glob(os.path.join(config.RAW_DIR, '*.xlsx')) + glob.glob(os.path.join(config.RAW_DIR, '*.csv')))
    return files[0] if files else None


def _require_hourly_cols(df, path):
    """Outdoor climate + A/C load are required. Indoor is modelled later, not loaded here."""
    missing = [c for c in config.REQUIRED_HOURLY_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Hourly dataset {os.path.basename(str(path))} missing required columns {missing}. "
            f"Need outdoor weather + ac_kwh after rename ({list(config.REQUIRED_HOURLY_COLS)}). "
            "Indoor temperature is not an input column; it is modelled in logic/indoor_model.py."
        )


def load_data(filepath=None, verbose=True):
    organizer = None if filepath else _find_organizer_file()
    if organizer:
        path, source, shift = organizer, 'organizer', 0
        raw = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
    else:
        path = filepath or config.DATA_FILE
        source, shift = config.DATA_SOURCE, config.OUTDOOR_PHASE_SHIFT_H
        raw = pd.read_excel(path, sheet_name='Hourly_Data')

    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    df = df.rename(columns=_RENAME)
    if source == 'organizer':
        df = df.rename(columns={k: v for k, v in _ORGANIZER_SYNONYMS.items() if k in df.columns and v not in df.columns})
    if 'timestamp' not in df.columns and 'date' in df.columns and 'hour' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'].astype(int), unit='h')
    _require_hourly_cols(df, path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    if 'date' not in df.columns:
        df['date'] = df['timestamp'].dt.normalize()
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'building_id' not in df.columns:
        df['building_id'] = 'ORG_001'
    if 'is_peak' not in df.columns:
        df['is_peak'] = df['hour'].isin(config.PEAK_HOURS).astype(int)

    df = _rotate_within_day(df, ['temp', 'humidity', 'dew_point'], shift)
    df['data_source'] = source
    df.attrs['provenance'] = {
        'data_source': source, 'phase_shift_h': shift, 'rows': int(len(df)),
        'buildings': int(df['building_id'].nunique()), 'file': os.path.basename(str(path)),
    }
    if verbose:
        print(f"Loaded {len(df)} rows from {df['timestamp'].min().date()} to {df['timestamp'].max().date()} "
              f"[{source}, phase shift {shift} h]")
    return df


def load_indoor_log(path):
    """Team's own DHT11 CSV (P1). Columns: timestamp, indoor_temp_c, indoor_rh_pct, outdoor_temp_c, ac_state(0/1).

    Used only for physics calibration (fit_tau), never as day-ahead RF features.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ('timestamp', 'indoor_temp_c', 'outdoor_temp_c', 'ac_state')
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Indoor DHT log missing {missing}; expected {list(need)}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['data_source'] = 'own_logger'
    df.attrs['provenance'] = {'data_source': 'own_logger', 'rows': int(len(df)), 'file': os.path.basename(path)}
    return df
