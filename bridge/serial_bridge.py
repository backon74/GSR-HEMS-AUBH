"""Laptop bridge: owns the simulated clock, sends one command per simulated hour, streams device telemetry.

    python bridge/serial_bridge.py --port /dev/cu.usbserial-0001      # real board
    python bridge/serial_bridge.py --dry-run                          # print lines, no hardware

The ESP32 does not simulate time. It reads the DHT continuously and reports each sample; this bridge
paces the hours and publishes every telemetry packet to data/processed/live_telemetry.json
(+ live_history.jsonl) so the dashboard is a live sensor view.

Start this BEFORE the board boots (the ESP32 resets when the port opens) and close the Arduino serial monitor.
If the bridge stops sending, the firmware falls back to firmware/replay_table.h. The device's local override
always outranks a command; the bridge only observes it via telemetry.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config  # noqa: E402
from dashboard.live_io import write_telemetry  # noqa: E402


def format_command(hour, mode, fan_pct, model_override=False):
    """H14,PEAK_REDUCE,75 (4th field ,MODEL when the model triggered an override)."""
    line = f"H{int(hour)},{config.MODE_TOKEN[mode]},{int(fan_pct)}"
    return line + ',MODEL' if (mode == 'comfort_override' and model_override) else line


def hour_seconds(hour):
    return config.SECONDS_PER_PEAK_HOUR if hour in config.PEAK_HOURS else config.SECONDS_PER_SIM_HOUR


def commands_from_demo(demo):
    return [(r['hour'], format_command(r['hour'], r['actual_mode'], r['fan_pwm'], r['override_reason'] == 'indoor_temp'), r)
            for r in demo['rows']]


def parse_telemetry(line):
    """Device JSON line -> dict, or None if malformed (never raises: a bad line must not stop the clock)."""
    try:
        d = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict) or d.get('override_source', 'none') not in config.OVERRIDE_SOURCES:
        return None
    return d


def snapshot(demo, row, tel=None):
    """One dashboard row: plan context from the schedule, sensor fields from the device."""
    live = tel is not None
    return {
        'source': 'esp32' if live else 'sim',
        'source_badge': 'LIVE' if live else 'SIM',
        'date': demo['date'],
        'sim_hour': int(row['hour']),
        'planned_mode': row['planned_mode'],
        'actual_mode': (tel or {}).get('actual_mode') or row['actual_mode'],
        'override_source': (tel or {}).get('override_source', 'none'),
        'temp': (tel or {}).get('temp'),
        'rh': (tel or {}).get('rh'),
        'dew': (tel or {}).get('dew'),
        'outdoor_temp': row.get('temp'),
        'outdoor_rh': row.get('humidity'),
        'outdoor_dew': row.get('dew_point'),
        'indoor_temp_est_c': row.get('indoor_temp_est_c'),
        'fan_pwm': (tel or {}).get('fan_pwm', row['fan_pwm']),
        'sensor_ok': (tel or {}).get('sensor_ok'),
        'kwh_baseline': row.get('kwh_baseline'),
        'kwh_optimized': row.get('kwh_optimized'),
    }


def run(port, demo_path, dry_run, speed=1.0, loops=1):
    demo = json.load(open(demo_path))
    cmds = commands_from_demo(demo)
    print(f"demo day {demo['date']} scenario={demo['scenario']['name'] if demo['scenario'] else 'none'} "
          f"data_source={demo['data_source']} loop={demo['loop_seconds']} s")
    print(f"telemetry -> {config.LIVE_TELEMETRY_PATH} (dashboard polls this; DHT samples stream when connected)")
    ser = None
    if not dry_run:
        import serial  # pyserial
        ser = serial.Serial(port, config.SERIAL_BAUD, timeout=0.2)
        time.sleep(2.0)
    try:
        for _ in range(loops):
            for hour, line, row in cmds:
                print('->', line)
                if ser:
                    ser.write((line + '\n').encode())
                write_telemetry(snapshot(demo, row), append_history=True)
                end = time.time() + hour_seconds(hour) / speed
                while time.time() < end:
                    if ser:
                        t = parse_telemetry(ser.readline().decode(errors='ignore').strip())
                        if t:
                            print('<-', json.dumps(t))
                            write_telemetry(snapshot(demo, row, t), append_history=True)
                    else:
                        time.sleep(min(0.05, max(0.0, end - time.time())))
    finally:
        if ser:
            ser.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port')
    ap.add_argument('--demo', default=config.DEMO_DAY_PATH)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--loops', type=int, default=1)
    a = ap.parse_args()
    if not a.dry_run and not a.port:
        ap.error('--port required unless --dry-run')
    run(a.port, a.demo, a.dry_run, a.speed, a.loops)
