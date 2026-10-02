"""Fake ESP32 DHT stream for dashboard work when no board is attached.

    python tools/fake_device_feed.py              # ~1 sample/s, walks the demo day
    python tools/fake_device_feed.py --once       # single packet

Writes the same telemetry rows the bridge would publish from real DHT packets, so the sensor
console can be reviewed end to end. Never used by pipeline.py and never feeds the ML model.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config  # noqa: E402
from bridge.serial_bridge import snapshot  # noqa: E402
from dashboard.live_io import write_telemetry  # noqa: E402


def magnus_dew(t, rh):
    g = math.log(max(rh, 1.0) / 100.0) + config.MAGNUS_A * t / (config.MAGNUS_B + t)
    return config.MAGNUS_B * g / (config.MAGNUS_A - g)


def fake_packet(row, phase):
    """Room temp drifts with the commanded mode; DHT11 reports integers."""
    pull = {'pre_cool': -1.6, 'peak_reduce': 1.1, 'comfort_override': 0.4, 'normal': 0.0}
    base = config.T_SET + pull.get(row['actual_mode'], 0.0) + 0.5 * math.sin(phase / 7.0)
    temp = round(base + random.uniform(-0.4, 0.4))
    rh = int(min(80, max(20, 48 + 9 * math.sin(phase / 11.0) + random.uniform(-2, 2))))
    return {
        'sim_hour': int(row['hour']),
        'temp': float(temp),
        'rh': rh,
        'dew': round(magnus_dew(temp, rh), 1),
        'planned_mode': row['planned_mode'],
        'actual_mode': row['actual_mode'],
        'override_source': 'model' if row['override_reason'] == 'indoor_temp' else 'none',
        'fan_pwm': int(row['fan_pwm']),
        'sensor_ok': True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', default=config.DEMO_DAY_PATH)
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--hour-seconds', type=float, default=4.0)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    demo = json.load(open(a.demo))
    rows = demo['rows']
    print(f"fake DHT feed -> {config.LIVE_TELEMETRY_PATH} (Ctrl-C to stop)")
    phase = 0
    while True:
        for row in rows:
            end = time.time() + a.hour_seconds
            while time.time() < end:
                write_telemetry(snapshot(demo, row, fake_packet(row, phase)), append_history=True)
                phase += 1
                if a.once:
                    return
                time.sleep(a.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
