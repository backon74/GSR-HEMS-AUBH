import math
import os
import re
import subprocess
import tempfile

import pytest

import config
from bridge import serial_bridge
from evaluation import exports


def test_maps_cover_all_modes():
    for m in config.MODES:
        assert m in config.MODE_TOKEN and m in config.MODE_LED and m in config.FAN_PWM and m in config.MODE_LCD
        assert config.MODE_LED[m] in config.LED_GPIO
    assert all(v >= config.FAN_FLOOR_PCT for v in config.FAN_PWM.values())


def test_gpios_unique():
    pins = list(config.LED_GPIO.values()) + [config.GPIO_BOOT]
    assert len(pins) == len(set(pins))


def test_loop_length_is_86s():
    assert exports.loop_seconds() == 86


def test_replay_table_equals_schedule_rows(fdf):
    demo = exports.select_demo_day(fdf)
    txt = exports.replay_table_text(demo)

    def arr(name):
        return [int(x) for x in re.search(name + r'\[24\] = \{([^}]*)\}', txt).group(1).split(',')]
    rows = demo['rows']
    idx = {m: i for i, m in enumerate(config.MODES)}
    assert arr('REPLAY_MODE') == [idx[r['actual_mode']] for r in rows]
    assert arr('REPLAY_FAN_PCT') == [r['fan_pwm'] for r in rows]
    assert arr('REPLAY_KWH_OPT_X100') == [round(r['kwh_optimized'] * 100) for r in rows]
    assert arr('REPLAY_HOUR_MS') == [r['hour_ms'] for r in rows]
    assert len(rows) == 24


def test_demo_day_shows_every_mode(fdf):
    demo = exports.select_demo_day(fdf)
    modes = {r['actual_mode'] for r in demo['rows']}
    assert {'normal', 'pre_cool', 'peak_reduce'} <= modes
    if demo['scenario']:
        assert demo['scenario_label'].startswith('SCENARIO')


def test_bridge_command_format():
    assert serial_bridge.format_command(14, 'peak_reduce', 75) == 'H14,PEAK_REDUCE,75'
    assert serial_bridge.format_command(15, 'comfort_override', 100, True) == 'H15,OVERRIDE,100,MODEL'
    assert serial_bridge.parse_telemetry('garbage') is None
    assert serial_bridge.parse_telemetry('{"override_source":"live","sim_hour":3}')['override_source'] == 'live'


@pytest.mark.skipif(not any(os.access(os.path.join(p, 'c++'), os.X_OK) for p in os.environ['PATH'].split(os.pathsep)),
                    reason='no C++ compiler')
def test_magnus_parity_python_vs_cpp():
    pts = [(25.0, 50.0), (30.0, 60.0), (40.0, 30.0), (14.0, 80.0), (27.8, 52.0)]
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'magnus.h'), 'w').write(exports.magnus_header_text())
        calls = ''.join(f'printf("%.4f\\n", magnus_dew({t}f,{h}f));' for t, h in pts)
        open(os.path.join(d, 'm.cpp'), 'w').write(f'#include <stdio.h>\n#include "magnus.h"\nint main(){{{calls}return 0;}}')
        subprocess.run(['c++', '-o', os.path.join(d, 'm'), os.path.join(d, 'm.cpp')], check=True, cwd=d)
        out = subprocess.run([os.path.join(d, 'm')], capture_output=True, text=True, check=True).stdout.split()
    for (t, h), c in zip(pts, out):
        assert abs(exports.magnus_dew(t, h) - float(c)) < 5e-3
    assert abs(exports.magnus_dew(25, 50) - 13.86) < 0.05
