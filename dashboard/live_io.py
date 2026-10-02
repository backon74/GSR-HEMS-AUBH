"""Shared live state between serial_bridge and the Streamlit dashboard.

The bridge publishes one row per DHT sample (plus one at each simulated hour change).
live_telemetry.json is the latest snapshot; live_history.jsonl is the rolling sample stream
the dashboard plots against wall-clock time.
"""
import json
import os
import time
from collections import deque
from datetime import datetime, timezone

import config

SENSOR_FIELDS = ('temp', 'rh', 'dew')


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def write_telemetry(payload, path=None, append_history=True):
    """Atomically write the latest snapshot; optionally append one JSONL history row."""
    path = path or config.LIVE_TELEMETRY_PATH
    _ensure_dir(path)
    row = dict(payload)
    row.setdefault('updated_at', datetime.now(timezone.utc).isoformat())
    row.setdefault('updated_unix', time.time())
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(row, f, indent=2, default=str)
    os.replace(tmp, path)
    if append_history:
        hist = config.LIVE_HISTORY_PATH
        _ensure_dir(hist)
        with open(hist, 'a') as f:
            f.write(json.dumps(row, default=str) + '\n')
    return row


def read_telemetry(path=None, max_age_s=None):
    """Return latest telemetry dict with age/fresh annotations, or None if missing/unreadable."""
    path = path or config.LIVE_TELEMETRY_PATH
    max_age_s = config.LIVE_TELEMETRY_STALE_S if max_age_s is None else max_age_s
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            row = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    age = time.time() - float(row.get('updated_unix') or 0)
    row['age_s'] = age
    row['fresh'] = age <= max_age_s
    row['has_sensor'] = any(row.get(k) is not None for k in SENSOR_FIELDS)
    return row


def load_history(path=None, limit=1200):
    """Tail of the sample stream, oldest first."""
    path = path or config.LIVE_HISTORY_PATH
    if not os.path.isfile(path):
        return []
    rows = deque(maxlen=limit)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return list(rows)


def feed_status(telemetry, history=None):
    """Summarise the sensor feed for the status bar."""
    if not telemetry:
        return {'state': 'offline', 'badge': 'NO FEED', 'age_s': None, 'samples': 0,
                'sensor_ok': None, 'rate_hz': None, 'source': None}
    sensor_rows = [r for r in (history or []) if any(r.get(k) is not None for k in SENSOR_FIELDS)]
    rate = None
    if len(sensor_rows) >= 2:
        span = float(sensor_rows[-1].get('updated_unix', 0)) - float(sensor_rows[0].get('updated_unix', 0))
        if span > 0:
            rate = round((len(sensor_rows) - 1) / span, 2)
    live = bool(telemetry.get('fresh') and telemetry.get('source') == 'esp32' and telemetry.get('has_sensor'))
    if live:
        state, badge = ('fault', 'SENSOR FAULT') if telemetry.get('sensor_ok') is False else ('live', 'LIVE')
    elif telemetry.get('fresh'):
        state, badge = 'replay', 'REPLAY'
    else:
        state, badge = 'stale', 'STALE'
    return {'state': state, 'badge': badge, 'age_s': telemetry.get('age_s'),
            'samples': len(sensor_rows), 'sensor_ok': telemetry.get('sensor_ok'),
            'rate_hz': rate, 'source': telemetry.get('source')}


def merge_device(payload, telemetry):
    """Fill get_payload()['device'] from ESP32 telemetry when present and fresh."""
    out = dict(payload)
    device = {k: dict(v) for k, v in payload.get('device', {}).items()}
    live = bool(telemetry and telemetry.get('fresh') and telemetry.get('source') == 'esp32'
                and telemetry.get('sensor_ok', True))
    if live:
        for dst, src in [('temp', 'temp'), ('rh', 'rh'), ('dew', 'dew'),
                         ('fan_duty', 'fan_pwm'), ('sensor_ok', 'sensor_ok')]:
            if telemetry.get(src) is not None:
                device[dst] = {'value': telemetry[src], 'tag': 'measured'}
        out['source_badge'] = 'LIVE'
    elif telemetry and telemetry.get('fresh') and telemetry.get('source') == 'sim':
        out['source_badge'] = telemetry.get('source_badge') or payload.get('source_badge') or 'SIM'
    out['device'] = device
    out['telemetry'] = telemetry
    return out


def snapshot_from_payload(payload, source='sim'):
    """Build a telemetry-shaped row from an engine payload (no DHT yet)."""
    m = payload['modelled']
    r = payload['replayed']
    return {
        'source': source,
        'source_badge': payload.get('source_badge', 'SIM'),
        'date': payload['date'],
        'sim_hour': payload['hour'],
        'planned_mode': m['planned_mode']['value'],
        'actual_mode': m['actual_mode']['value'],
        'override_source': 'model' if m['override_reason']['value'] != 'none' else 'none',
        'temp': None,
        'rh': None,
        'dew': None,
        'outdoor_temp': r['outdoor_temp']['value'],
        'outdoor_rh': r['outdoor_rh']['value'],
        'outdoor_dew': r['outdoor_dew']['value'],
        'indoor_temp_est_c': m['indoor_temp_est_c']['value'],
        'fan_pwm': m['fan_pwm']['value'],
        'sensor_ok': None,
        'kwh_baseline': r['kwh_baseline']['value'],
        'kwh_optimized': m['kwh_optimized']['value'],
    }
