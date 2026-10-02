"""Live telemetry I/O for dashboard ↔ bridge."""
import json
import os
import time

import config
from dashboard.live_io import merge_device, read_telemetry, write_telemetry
from logic.engine import get_payload


def test_write_and_read_telemetry(tmp_path, monkeypatch):
    path = tmp_path / 'live_telemetry.json'
    hist = tmp_path / 'live_history.jsonl'
    monkeypatch.setattr(config, 'LIVE_TELEMETRY_PATH', str(path))
    monkeypatch.setattr(config, 'LIVE_HISTORY_PATH', str(hist))
    write_telemetry({'source': 'sim', 'sim_hour': 3, 'temp': None, 'rh': None, 'dew': None},
                    append_history=True)
    row = read_telemetry(max_age_s=30)
    assert row and row['fresh'] and row['sim_hour'] == 3 and row['source'] == 'sim'
    assert hist.is_file() and 'sim_hour' in hist.read_text()


def test_merge_device_promotes_esp32_to_live(monkeypatch, tmp_path):
    path = tmp_path / 'live_telemetry.json'
    monkeypatch.setattr(config, 'LIVE_TELEMETRY_PATH', str(path))
    write_telemetry({
        'source': 'esp32', 'sim_hour': 14, 'temp': 27.5, 'rh': 55, 'dew': 17.0,
        'fan_pwm': 100, 'sensor_ok': True,
    }, append_history=False)
    # freshen mtime/unix
    tel = read_telemetry(max_age_s=30)
    payload = {
        'source_badge': 'SIM',
        'device': {k: {'value': None, 'tag': 'measured'} for k in ('temp', 'rh', 'dew', 'fan_duty', 'sensor_ok')},
    }
    out = merge_device(payload, tel)
    assert out['source_badge'] == 'LIVE'
    assert out['device']['temp']['value'] == 27.5
    assert out['device']['temp']['tag'] == 'measured'


def test_laptop_owns_clock():
    from bridge import serial_bridge
    assert config.CLOCK_OWNER == 'laptop'
    demo = json.load(open(config.DEMO_DAY_PATH))
    cmds = serial_bridge.commands_from_demo(demo)
    assert len(cmds) == 24
    assert cmds[12][1].startswith('H12,')


def test_feed_status_distinguishes_live_replay_offline(monkeypatch, tmp_path):
    from dashboard.live_io import feed_status
    assert feed_status(None)['state'] == 'offline'
    now = time.time()
    sim = {'fresh': True, 'source': 'sim', 'has_sensor': False, 'age_s': 0.2}
    assert feed_status(sim)['state'] == 'replay'
    live_rows = [{'temp': 27.0, 'updated_unix': now - 2}, {'temp': 27.2, 'updated_unix': now}]
    live = {'fresh': True, 'source': 'esp32', 'has_sensor': True, 'sensor_ok': True, 'age_s': 0.3}
    s = feed_status(live, live_rows)
    assert s['state'] == 'live' and s['samples'] == 2 and s['rate_hz'] == 0.5
    fault = dict(live, sensor_ok=False)
    assert feed_status(fault, live_rows)['state'] == 'fault'
    assert feed_status({'fresh': False, 'source': 'esp32', 'has_sensor': True})['state'] == 'stale'


def test_get_payload_has_device_slots():
    demo = json.load(open(config.DEMO_DAY_PATH))
    sc = demo.get('scenario')
    scenario = config.SCENARIOS.get(sc['name']) if sc else None
    p = get_payload(demo['date'], 12, scenario=scenario)
    assert p['modelled']['actual_mode']['value'] in config.MODES
    assert p['device']['temp']['value'] is None
    assert p['replayed']['outdoor_temp']['value'] is not None
