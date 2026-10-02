"""Laptop bridge: owns the simulated clock, sends one command line per simulated hour, logs device telemetry.

    python bridge/serial_bridge.py --port /dev/cu.usbserial-0001      # real board
    python bridge/serial_bridge.py --dry-run                          # print lines, no hardware

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


def format_command(hour, mode, fan_pct, model_override=False):
    """H14,PEAK_REDUCE,75 (4th field ,MODEL when the model triggered an override)."""
    line = f"H{int(hour)},{config.MODE_TOKEN[mode]},{int(fan_pct)}"
    return line + ',MODEL' if (mode == 'comfort_override' and model_override) else line


def hour_seconds(hour):
    return config.SECONDS_PER_PEAK_HOUR if hour in config.PEAK_HOURS else config.SECONDS_PER_SIM_HOUR


def commands_from_demo(demo):
    return [(r['hour'], format_command(r['hour'], r['actual_mode'], r['fan_pwm'], r['override_reason'] == 'indoor_temp'))
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


def run(port, demo_path, dry_run, speed=1.0, loops=1):
    demo = json.load(open(demo_path))
    cmds = commands_from_demo(demo)
    print(f"demo day {demo['date']} scenario={demo['scenario']['name'] if demo['scenario'] else 'none'} "
          f"data_source={demo['data_source']} loop={demo['loop_seconds']} s")
    ser = None
    if not dry_run:
        import serial  # pyserial
        ser = serial.Serial(port, config.SERIAL_BAUD, timeout=0.2)
        time.sleep(2.0)
    for _ in range(loops):
        for hour, line in cmds:
            print('->', line)
            if ser:
                ser.write((line + '\n').encode())
            end = time.time() + hour_seconds(hour) / speed
            while time.time() < end:
                if ser:
                    t = parse_telemetry(ser.readline().decode(errors='ignore').strip())
                    if t:
                        print('<-', json.dumps(t))
                else:
                    time.sleep(min(0.05, max(0.0, end - time.time())))
    if ser:
        ser.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port')
    ap.add_argument('--demo', default='results/demo_day.json')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--loops', type=int, default=1)
    a = ap.parse_args()
    if not a.dry_run and not a.port:
        ap.error('--port required unless --dry-run')
    run(a.port, a.demo, a.dry_run, a.speed, a.loops)
