"""Sky and dial rendering: the solar model, the palette ramp and the hand angles.

The solar numbers below are checked against the real sun over config.SITE_LAT / SITE_LON,
so a regression in the declination or hour-angle maths fails here rather than showing up as
a sunlit midnight on the dashboard.
"""
import datetime
import re

import config
from dashboard.skyclock import _orb_xy, _phase, dial, sky, solar_position

DEMO_DOY = datetime.date(2025, 7, 25).timetuple().tm_yday   # 206


def _elevation(hour, doy=DEMO_DOY):
    return solar_position(doy, hour)[0]


def test_solar_noon_is_high_and_midnight_is_far_below():
    # Dammam sits at 26.4 N, so late-July noon is within a few degrees of overhead.
    assert 78.0 < _elevation(11.7) <= 90.0
    assert _elevation(0.0) < -35.0


def test_sunrise_and_sunset_bracket_the_real_times():
    """Late-July sunrise is about 05:10 and sunset about 18:40 local."""
    rise = next(h / 60.0 for h in range(4 * 60, 8 * 60) if _elevation(h / 60.0) > -0.833)
    sets = next(h / 60.0 for h in range(16 * 60, 22 * 60) if _elevation(h / 60.0) < -0.833)
    assert 4.8 < rise < 5.5, rise
    assert 18.3 < sets < 19.0, sets
    assert 13.0 < sets - rise < 13.9, 'July day length in the Gulf is a little over 13 h'


def test_azimuth_runs_east_to_west():
    assert solar_position(DEMO_DOY, 6.0)[1] < 110.0      # morning sun in the east
    assert solar_position(DEMO_DOY, 18.0)[1] > 250.0     # evening sun in the west


def test_sky_is_dark_at_night_and_lit_by_day():
    night = sky(DEMO_DOY, 1.0)
    noon = sky(DEMO_DOY, 12.0)
    assert night['darkness'] == 1.0 and not night['sun_up']
    assert noon['darkness'] == 0.0 and noon['sun_up']
    # Night zenith must actually be darker than the daylight zenith.
    assert sum(int(night['zenith'][i:i + 2], 16) for i in (1, 3, 5)) < \
           sum(int(noon['zenith'][i:i + 2], 16) for i in (1, 3, 5))


def test_sky_colours_are_hex_and_interpolated():
    for h in (0, 3, 5, 5.5, 7, 12, 17, 18.5, 19, 22):
        s = sky(DEMO_DOY, h)
        for k in ('zenith', 'mid', 'horizon', 'orb'):
            assert re.fullmatch(r'#[0-9a-f]{6}', s[k]), (h, k, s[k])


def test_phase_names_follow_elevation():
    assert _phase(80, True) == 'high sun'
    assert _phase(0.5, True) == 'sunrise'
    assert _phase(0.5, False) == 'sunset'
    assert _phase(-3, True) == 'dawn'
    assert _phase(-3, False) == 'dusk'
    assert _phase(-40, False) == 'night'


def test_moon_is_up_when_the_sun_is_down():
    s = sky(DEMO_DOY, 23.0)
    assert not s['sun_up']
    assert s['moon_elevation'] > 0, 'anti-solar point should be above the horizon at midnight'


def test_orb_stays_in_the_open_sky_band():
    """The plates are bottom-aligned and opaque, so the orb must stay in the top third."""
    for el in (-6, 0, 10, 45, 82):
        for az in (60, 90, 180, 270, 300):
            x, y = _orb_xy(el, az)
            assert -5 <= x <= 105, (el, az, x)
            assert 4 <= y <= 36, (el, az, y)


def test_live_dial_hand_angles_match_the_clock():
    # 15:42:30 -> hour 3h42m30s past noon, minute 42.5 min, second 30 s.
    secs = 15 * 3600 + 42 * 60 + 30
    svg = dial('live', seconds_into_day=secs, caption='x', readout='15:42')
    got = [float(a) for a in re.findall(r'rotate\(([-\d.]+)deg\)', svg)]
    assert len(got) == 3
    assert abs(got[0] - (secs % 43200) / 43200 * 360) < 0.01      # 111.375
    assert abs(got[1] - 42.5 / 60 * 360) < 0.01                   # 255.0
    assert abs(got[2] - 180.0) < 0.01


def test_step_dial_puts_the_hour_hand_on_the_hour():
    for hour, deg in ((0, 0.0), (3, 90.0), (13, 30.0), (18, 180.0)):
        svg = dial('step', hour=hour, caption='c', readout=f'{hour:02d}:00')
        angles = [float(a) for a in re.findall(r'rotate\(([-\d.]+)deg\)', svg)]
        assert angles and abs(angles[0] - deg) < 0.01, (hour, angles)


def test_site_constants_present():
    for name in ('SITE_LABEL', 'SITE_LAT', 'SITE_LON', 'SITE_TZ_OFFSET_H'):
        assert hasattr(config, name)
