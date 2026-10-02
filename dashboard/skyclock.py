"""Sky and dial rendering for the SmartCool console.

The sun on screen is the real sun over the site in config (SITE_LAT / SITE_LON): solar
declination and hour angle give its elevation and azimuth, and every colour, the star
opacity and the orb position are derived from that elevation. So the hero sky at 05:00 is
dawn because the sun is actually 2 degrees up, not because an hour number was mapped to a
palette.

Two clocks use this module. The wall-clock dial sweeps continuously in CSS (negative
animation-delay seeds it to the current second, so it needs no Streamlit rerun). The
simulated-day dial is a stepper: Python owns the hour, the hand steps to it.
"""
import math
import random

import config

# elevation deg, zenith, mid, horizon, orb, label
_SKY_STOPS = (
    (-18.0, '#03050b', '#070b16', '#0c1320', '#d5e2f2', 'night'),
    (-12.0, '#04081a', '#0a1226', '#141f36', '#dde7f4', 'astronomical twilight'),
    (-6.0, '#0a1536', '#1b2748', '#3a3a5e', '#f6e6cc', 'nautical twilight'),
    (-0.9, '#16305c', '#46406c', '#9a5048', '#ffd6a0', 'civil twilight'),
    (0.0, '#1d3d6b', '#63507a', '#d96b36', '#ffbe72', 'horizon'),
    (6.0, '#2a5f95', '#85899f', '#eda75c', '#ffd796', 'golden hour'),
    (18.0, '#2d76b0', '#86aecb', '#e6d2b0', '#fff1d2', 'full light'),
    (40.0, '#2b78b6', '#94bed8', '#e2ebf0', '#fff8e6', 'daylight'),
    (70.0, '#2878b8', '#9cc6dd', '#edf3f5', '#fffdf0', 'high sun'),
)


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _mix(a, b, t):
    ra, rb = _hex_to_rgb(a), _hex_to_rgb(b)
    return '#%02x%02x%02x' % tuple(round(ra[i] + (rb[i] - ra[i]) * t) for i in range(3))


def solar_position(doy, local_hour, lat=None, lon=None, tz_h=None):
    """Sun elevation and azimuth (deg) for a day-of-year and local decimal hour.

    Cooper declination plus the Spencer equation of time: about a minute of arc, which is
    far finer than anything the sky gradient can show.
    """
    lat = config.SITE_LAT if lat is None else lat
    lon = config.SITE_LON if lon is None else lon
    tz_h = config.SITE_TZ_OFFSET_H if tz_h is None else tz_h
    b = math.radians(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)   # minutes
    solar_time = local_hour + (4.0 * (lon - 15.0 * tz_h) + eot) / 60.0
    ha = math.radians(15.0 * (solar_time - 12.0))
    decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + doy) / 365.0)))
    phi = math.radians(lat)
    sin_el = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    el = math.asin(max(-1.0, min(1.0, sin_el)))
    cos_el, cos_phi = math.cos(el), math.cos(phi)
    if abs(cos_el * cos_phi) < 1e-9:
        az = 180.0
    else:
        cos_az = (math.sin(decl) - math.sin(el) * math.sin(phi)) / (cos_el * cos_phi)
        az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
        if ha > 0:
            az = 360.0 - az
    return math.degrees(el), az


def _phase(el, rising):
    """Name the light the way someone outdoors would, from solar elevation."""
    if el > 55:
        return 'high sun'
    if el > 30:
        return 'daylight'
    if el > 12:
        return 'full light'
    if el > 3:
        return 'golden hour'
    if el > -0.833:
        return 'sunrise' if rising else 'sunset'
    if el > -6:
        return 'dawn' if rising else 'dusk'
    if el > -12:
        return 'nautical twilight'
    if el > -18:
        return 'astronomical twilight'
    return 'night'


def sky(doy, local_hour):
    """Everything the sky layer needs for one instant."""
    el, az = solar_position(doy, local_hour)
    lo = _SKY_STOPS[0]
    hi = _SKY_STOPS[-1]
    for i in range(len(_SKY_STOPS) - 1):
        if _SKY_STOPS[i][0] <= el <= _SKY_STOPS[i + 1][0]:
            lo, hi = _SKY_STOPS[i], _SKY_STOPS[i + 1]
            break
    else:
        lo = hi = _SKY_STOPS[0] if el < _SKY_STOPS[0][0] else _SKY_STOPS[-1]
    span = hi[0] - lo[0]
    t = 0.0 if span <= 0 else max(0.0, min(1.0, (el - lo[0]) / span))
    label = _phase(el, az < 180.0)
    # Night deepens from first dusk to astronomical dark; drives stars and the moon.
    darkness = max(0.0, min(1.0, (-el - 1.0) / 15.0))
    up = el > -0.833
    # Moon at the anti-solar point: exact for a full moon, and it keeps the disc on a real
    # arc instead of a decorative spot. Lunar phase is not modelled.
    m_el, m_az = solar_position(doy, local_hour + 12.0)
    return {
        'elevation': el, 'azimuth': az, 'label': label, 'darkness': darkness, 'sun_up': up,
        'moon_elevation': m_el, 'moon_azimuth': m_az,
        'zenith': _mix(lo[1], hi[1], t), 'mid': _mix(lo[2], hi[2], t),
        'horizon': _mix(lo[3], hi[3], t), 'orb': _mix(lo[4], hi[4], t),
    }


def _orb_xy(el, az):
    """Project the orb onto the sky band: azimuth across, elevation up.

    Elevation is mapped into the top third only. The instrument plates are bottom-aligned
    and opaque, so a literal horizon-to-zenith mapping would park the sun behind them at
    exactly the hours the sky matters most. Azimuth stays true.
    """
    x = max(-4.0, min(104.0, (az - 55.0) / 250.0 * 100.0))
    y = 33.0 - max(-6.0, min(78.0, el)) / 78.0 * 27.0
    return round(x, 2), round(y, 2)


_STARS = None


def _stars():
    global _STARS
    if _STARS is None:
        rng = random.Random(7)
        # Thinned towards the horizon, the way haze actually eats the faint ones.
        _STARS = []
        for _ in range(150):
            y = round(rng.triangular(0, 78, 12), 2)
            _STARS.append((round(rng.uniform(0, 100), 2), y,
                           round(rng.uniform(0.9, 2.4), 2),
                           round(rng.uniform(0.45, 1.0) * (1.0 - y / 110.0), 3),
                           round(rng.uniform(0, 5), 2)))
    return _STARS


def sky_layer(s, orb_label=''):
    """Background atmosphere: gradient, star field, glow, sun or moon disc, horizon haze."""
    el, az = s['elevation'], s['azimuth']
    if s['sun_up']:
        ox, oy = _orb_xy(el, az)
        orb = (f"<div class='sc-orb sc-sun' style='left:{ox}%;top:{oy}%;background:{s['orb']};'></div>"
               f"<div class='sc-glow' style='left:{ox}%;top:{oy}%;"
               f"background:radial-gradient(circle,{s['orb']}55 0%,{s['orb']}00 68%)'></div>")
    elif s['moon_elevation'] > -2.0:
        # Full disc, no terminator: the moon is drawn at the anti-solar point, where it is
        # by definition full. Shading it would assert a phase this module does not model.
        mx, my = _orb_xy(s['moon_elevation'], s['moon_azimuth'])
        a = round(0.3 + 0.65 * s['darkness'], 3)
        orb = (f"<div class='sc-orb sc-moon' style='left:{mx}%;top:{my}%;opacity:{a}'></div>"
               f"<div class='sc-glow' style='left:{mx}%;top:{my}%;opacity:{a};"
               f"background:radial-gradient(circle,#cfe0ff38 0%,#cfe0ff00 58%)'></div>")
    else:
        orb = ''
    stars = ''.join(
        f"<i style='left:{x}%;top:{y}%;width:{r}px;height:{r}px;"
        f"opacity:{round(o * s['darkness'], 3)};animation-delay:-{d}s'></i>"
        for x, y, r, o, d in _stars()
    ) if s['darkness'] > 0.02 else ''
    return (
        f"<div class='sc-sky' style=\"background:linear-gradient(180deg,{s['zenith']} 0%,"
        f"{s['mid']} 52%,{s['horizon']} 100%)\">"
        f"<div class='sc-stars'>{stars}</div><div class='sc-orbit'>{orb}</div>"
        f"<div class='sc-airglow' style='opacity:{round(s['darkness'] * 0.85, 3)}'></div>"
        f"<div class='sc-haze' style='background:linear-gradient(180deg,{s['horizon']}00,{s['horizon']}cc)'></div>"
        f"<div class='sc-ground' style=\"background:linear-gradient(180deg,{_mix(s['horizon'], '#05080b', 0.72)} 0%,"
        f"{_mix(s['horizon'], '#05080b', 0.94)} 100%);"
        f"border-top-color:{_mix(s['horizon'], '#ffffff', 0.18)}33\"></div>"
        f"<div class='sc-skylabel'>{orb_label or config.SITE_LABEL}"
        f"<span>{s['label']} · sun {el:+.1f}°</span></div></div>"
    )


def _face(numerals=True):
    """Tick ring and numerals, cut into the face rather than printed on it.

    Each mark is drawn twice: a dark stroke offset down-right, then the light stroke on top.
    That offset pair is what reads as incised metal instead of flat ink.
    """
    out = []
    for i in range(60):
        a = math.radians(i * 6.0)
        major = i % 5 == 0
        r1, w = (40.0, 1.9) if major else (43.5, 0.85)
        x1, y1 = 50 + r1 * math.sin(a), 50 - r1 * math.cos(a)
        x2, y2 = 50 + 46.5 * math.sin(a), 50 - 46.5 * math.cos(a)
        out.append(f"<line x1='{x1 + 0.4:.2f}' y1='{y1 + 0.4:.2f}' x2='{x2 + 0.4:.2f}' "
                   f"y2='{y2 + 0.4:.2f}' stroke-width='{w}' class='sc-cut'/>")
        out.append(f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
                   f"stroke-width='{w}' class='{'sc-tick-maj' if major else 'sc-tick'}'/>")
    if numerals:
        for n, ang in ((12, 0), (3, 90), (6, 180), (9, 270)):
            a = math.radians(ang)
            x, y = 50 + 31.5 * math.sin(a), 50 - 31.5 * math.cos(a)
            out.append(f"<text x='{x + 0.35:.2f}' y='{y + 0.35:.2f}' class='sc-num sc-num-cut'>{n}</text>")
            out.append(f"<text x='{x:.2f}' y='{y:.2f}' class='sc-num'>{n}</text>")
    return ''.join(out)


def dial(mode='live', hour=0.0, seconds_into_day=0.0, caption='', readout='', accent='#c9a227'):
    """Analog dial.

    Every hand angle is computed here, in Python, so the dial can never disagree with the
    digital readout beside it. CSS animation was the obvious way to sweep the hands without a
    rerun, but Streamlit reuses the markdown DOM node, which leaves a running animation's
    start time intact while animation-delay is replaced: the hands then drift ahead of the
    clock by however long the node has been mounted. Streamlit reruns once a second, which is
    exactly the cadence a quartz movement ticks at, so stepping the hands server-side is both
    simpler and honest.
    """
    if mode == 'live':
        sec = seconds_into_day % 86400.0
        hands = (
            f"<g style='transform:rotate({sec % 43200 / 43200 * 360:.3f}deg)'>"
            f"<path d='M50 56 L50 24' class='sc-h'/></g>"
            f"<g style='transform:rotate({sec % 3600 / 3600 * 360:.3f}deg)'>"
            f"<path d='M50 59 L50 13' class='sc-m'/></g>"
            f"<g style='transform:rotate({sec % 60 / 60 * 360:.3f}deg)'>"
            f"<path d='M50 64 L50 11' class='sc-s' stroke='{accent}'/>"
            f"<circle cx='50' cy='62' r='2.6' fill='{accent}'/></g>"
        )
    else:
        # Whole simulated hours only, so the minute hand stays at twelve by design.
        hands = (
            f"<g class='sc-step' style='transform:rotate({(hour % 12) * 30.0:.2f}deg)'>"
            f"<path d='M50 56 L50 24' class='sc-h'/></g>"
            f"<g><path d='M50 59 L50 13' class='sc-m'/></g>"
        )
    # The caption and readout live on a nameplate under the bezel, not on the dial face.
    # No radius inside the numeral ring escapes the hands: the minute hand reaches r=37 and
    # the hour hand r=26, so anything printed there gets struck through at common angles.
    plate_bits = ''
    if caption:
        plate_bits += f"<span class='sc-np-cap'>{caption}</span>"
    if readout:
        plate_bits += f"<span class='sc-np-read' style='color:{accent}'>{readout}</span>"
    nameplate = f"<div class='sc-nameplate'>{plate_bits}</div>" if plate_bits else ''
    svg = (
        f"<svg class='sc-dial' viewBox='0 0 100 100' role='img' aria-label='{caption} {readout}'>"
        f"<defs>"
        f"<radialGradient id='bez' cx='38%' cy='26%'>"
        f"<stop offset='0%' stop-color='#6b7681'/><stop offset='52%' stop-color='#2c343c'/>"
        f"<stop offset='100%' stop-color='#596570'/></radialGradient>"
        f"<radialGradient id='fce' cx='42%' cy='22%'>"
        f"<stop offset='0%' stop-color='#1b2229'/><stop offset='100%' stop-color='#0a0e12'/>"
        f"</radialGradient></defs>"
        f"<circle cx='50' cy='50' r='49.2' fill='url(#bez)'/>"
        f"<circle cx='50' cy='50' r='45.6' fill='url(#fce)' stroke='#0a0e12' stroke-width='0.7'/>"
        f"<circle cx='50' cy='50' r='44' fill='none' stroke='#ffffff14' stroke-width='0.5'/>"
        f"{_face()}{hands}"
        f"<circle cx='50' cy='50' r='3.1' fill='#4a5560'/>"
        f"<circle cx='50' cy='50' r='1.5' fill='{accent}'/>"
        f"</svg>"
    )
    return f"<div class='sc-dialwrap'>{svg}{nameplate}</div>"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Azeret+Mono:wght@400;500;600&display=swap');

:root {
  /* 0.92 alpha, not 0.80: the plates have to hold 4.5:1 for 0.66rem labels against a
     bleached-haze sky at 82 degrees as well as against midnight. */
  --sc-plate: rgba(12,17,21,0.92);
  --sc-plate-solid: #0d1217;
  --sc-bezel: rgba(255,255,255,0.14);
  --sc-ink: #eef3f7;
  --sc-ink-2: #b6c4cf;
  --sc-ink-3: #9dabb6;
  --sc-brass: #c9a227;
}
html, body, [class*="css"] { font-family: 'Archivo', system-ui, sans-serif; }
.stApp { background: #05080b; color: var(--sc-ink); }
.block-container { padding: 0 0 4rem !important; max-width: 100% !important; }
/* Kill Streamlit chrome; keep only the sidebar expand control as an instrument latch. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stDeployButton { display: none !important; }
header[data-testid="stHeader"] {
  background: transparent !important; border: 0 !important;
  height: 0 !important; min-height: 0 !important; padding: 0 !important;
}
[data-testid="stExpandSidebarButton"] {
  position: fixed !important; top: 0.85rem; left: 0.85rem; z-index: 1000;
  width: 2rem !important; height: 2rem !important;
  background: linear-gradient(180deg, #202830 0%, #141a20 100%) !important;
  border: 1px solid rgba(255,255,255,0.14) !important; border-radius: 2px !important;
  box-shadow: 0 6px 16px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08) !important;
  color: var(--sc-ink-2) !important;
}
[data-testid="stExpandSidebarButton"]:hover {
  border-color: rgba(201,162,39,0.55) !important; color: var(--sc-brass) !important;
}
::selection { background: rgba(201,162,39,0.32); color: #fff; }
::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: #05080b; }
::-webkit-scrollbar-thumb { background: #27313a; border: 3px solid #05080b; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #3a4752; }
:focus-visible { outline: 2px solid var(--sc-brass); outline-offset: 2px; }

/* Sidebar = service panel, same graphite/brass language as the plates. */
.stApp [data-testid="stSidebar"] {
  background: #080c10 !important;
  border-right: 1px solid rgba(255,255,255,0.07) !important;
  color: var(--sc-ink-2) !important;
}
[data-testid="stSidebar"] > div:first-child {
  background: #080c10 !important;
  padding-top: 1.1rem !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] .stCaption {
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.68rem !important; line-height: 1.45 !important;
  color: var(--sc-ink-3) !important; letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] [data-testid="stCode"],
[data-testid="stSidebar"] pre {
  background: #070b0e !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 3px !important;
  color: #9fb0bd !important;
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.66rem !important;
}
[data-testid="stSidebar"] hr {
  border: 0 !important; border-top: 1px solid rgba(255,255,255,0.09) !important;
  margin: 1rem 0 !important;
}
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  background: linear-gradient(180deg, #202830 0%, #141a20 100%) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 2px !important;
  color: var(--sc-ink) !important;
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.66rem !important; letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  box-shadow: 0 6px 16px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
  border-color: rgba(201,162,39,0.55) !important; color: var(--sc-brass) !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label {
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.64rem !important; letter-spacing: 0.16em !important;
  text-transform: uppercase !important; color: var(--sc-ink-3) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
  background: var(--sc-brass) !important;
  border: 1px solid #a8861a !important;
  box-shadow: 0 0 0 2px rgba(201,162,39,0.18) !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="presentation"] > div {
  background: rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebarCollapseButton"] {
  color: var(--sc-ink-2) !important;
}
[data-testid="stSidebarCollapseButton"]:hover {
  color: var(--sc-brass) !important;
}

/* ── sky band ───────────────────────────────────────────────────────────── */
/* The hero owns the whole first viewport: at 66vh the second band intruded on load and the
   page opened on two dials and two moons. */
.sc-stage { position: relative; min-height: 96vh; overflow: hidden; }
.sc-stage.sc-short { min-height: 62vh; }
.sc-sky { position: absolute; inset: 0; }
.sc-stars i {
  position: absolute; background: #fff; border-radius: 50%;
  animation: sc-twinkle 4.5s ease-in-out infinite;
}
/* Scale only. Touching opacity here would override the per-star inline value that carries
   both the horizon thinning and the darkness fade. */
@keyframes sc-twinkle { 0%, 100% { transform: scale(1) } 50% { transform: scale(0.5) } }
.sc-orb { position: absolute; width: 58px; height: 58px; margin: -29px 0 0 -29px; border-radius: 50%; }
.sc-sun { box-shadow: 0 0 38px 9px rgba(255,236,190,0.42); }
/* Full disc, but not a flat dot: limb darkening plus two faint maria so the material still
   reads as rock at low opacity. No terminator — the anti-solar point is full by definition. */
.sc-moon {
  background:
    radial-gradient(circle at 38% 32%, rgba(120,132,150,0.30) 0%, rgba(120,132,150,0) 26%),
    radial-gradient(circle at 64% 62%, rgba(120,132,150,0.22) 0%, rgba(120,132,150,0) 20%),
    radial-gradient(circle at 42% 36%, #f4f7fc 0%, #dce4ef 58%, #aab6c8 100%);
  box-shadow: 0 0 24px 5px rgba(207,224,255,0.26);
}
/* The containing block for the sun and moon. _orb_xy maps elevation into the top third of
   THIS box, not of the band — the band's height is content-driven, so on a narrow screen
   with four stacked plates a near-horizon orb otherwise landed inside the dial. */
.sc-orbit { position: absolute; left: 0; right: 0; top: 0; height: 100%; }
.sc-glow { position: absolute; width: 480px; height: 480px; margin: -240px 0 0 -240px; pointer-events: none; }
.sc-airglow {
  position: absolute; inset: 0; pointer-events: none;
  background:
    radial-gradient(130% 62% at 50% 100%, rgba(72,104,140,0.30) 0%, rgba(72,104,140,0) 62%),
    radial-gradient(60% 40% at 18% 8%, rgba(96,120,190,0.14) 0%, rgba(96,120,190,0) 70%);
}
.sc-haze { position: absolute; left: 0; right: 0; top: 48%; bottom: 13%; }
.sc-ground {
  position: absolute; left: 0; right: 0; bottom: 0; height: 13%;
  border-top: 1px solid transparent;   /* both tinted from the horizon by sky_layer() */
}
.sc-skylabel {
  position: absolute; left: 2.4rem; bottom: 1.1rem; z-index: 3;
  font-family: 'Azeret Mono', monospace; font-size: 0.7rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--sc-ink-2);
}
.sc-skylabel span { color: var(--sc-ink-3); letter-spacing: 0.08em; }
.sc-skylabel span::before { content: '·'; margin: 0 0.6rem; opacity: 0.55; }

/* ── dial ───────────────────────────────────────────────────────────────── */
.sc-dialwrap { display: grid; justify-items: center; gap: 0.9rem; }
.sc-dial { width: 100%; max-width: 372px; filter: drop-shadow(0 26px 44px rgba(0,0,0,0.55)); }
.sc-dial .sc-tick { stroke: #7f8d99; }
.sc-dial .sc-tick-maj { stroke: #dde5ec; }
.sc-dial .sc-cut { stroke: #04070a; opacity: 0.85; }
.sc-num {
  fill: #cfd9e2; font-family: 'Archivo', sans-serif; font-size: 8.4px; font-weight: 600;
  text-anchor: middle; dominant-baseline: central; letter-spacing: -0.02em;
}
.sc-num-cut { fill: #04070a; opacity: 0.8; }
/* Engraved nameplate under the bezel — the only place on the dial the hands cannot reach. */
.sc-nameplate {
  display: inline-flex; align-items: baseline; gap: 0.85rem;
  background: linear-gradient(180deg, #202830 0%, #141a20 100%);
  border: 1px solid rgba(255,255,255,0.1); border-radius: 2px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
  padding: 0.4rem 0.95rem;
}
.sc-np-cap {
  font-family: 'Azeret Mono', monospace; font-size: 0.64rem; letter-spacing: 0.19em;
  text-transform: uppercase; color: #93a2ae;
}
.sc-np-read {
  font-family: 'Azeret Mono', monospace; font-size: 1.1rem; font-weight: 600;
  font-variant-numeric: tabular-nums; letter-spacing: -0.02em;
}
.sc-h, .sc-m, .sc-s { stroke-linecap: round; }
.sc-h { stroke: #eff4f8; stroke-width: 4.4; }
.sc-m { stroke: #cdd7e0; stroke-width: 2.5; }
.sc-s { stroke-width: 1.1; }
.sc-dial g { transform-origin: 50px 50px; }
.sc-step { transition: transform 420ms cubic-bezier(0.22, 1, 0.36, 1); }
@media (prefers-reduced-motion: reduce) {
  .sc-stars i { animation: none; }
  .sc-step { transition: none; }
}

/* ── instrument plates ──────────────────────────────────────────────────── */
.sc-wrap {
  position: relative; z-index: 2; display: grid; gap: 1.7rem;
  grid-template-columns: minmax(290px, 404px) minmax(320px, 700px);
  justify-content: start; align-items: end;
  padding: 9rem 2.4rem 3.6rem; min-height: inherit;
}
.sc-stack { display: grid; gap: 0.7rem; }
.sc-plate {
  background: var(--sc-plate); border: 1px solid var(--sc-bezel); border-radius: 3px;
  box-shadow: 0 14px 30px rgba(0,0,0,0.42), inset 0 1px 0 rgba(255,255,255,0.07);
  backdrop-filter: blur(7px); padding: 0.9rem 1.15rem;
}
.sc-plate.sc-lead { padding: 1.25rem 1.4rem 1.1rem; }
.sc-lbl {
  font-family: 'Azeret Mono', monospace; font-size: 0.66rem; letter-spacing: 0.17em;
  text-transform: uppercase; color: var(--sc-ink-3);
}
.sc-val {
  font-family: 'Azeret Mono', monospace; font-weight: 600; font-variant-numeric: tabular-nums;
  line-height: 1; letter-spacing: -0.03em; color: var(--sc-ink); margin-top: 0.3rem;
}
.sc-val.sc-xl { font-size: 4.3rem; }
.sc-val.sc-lg { font-size: 2.3rem; }
.sc-val.sc-md { font-size: 1.5rem; }
.sc-val u { font-size: 0.4em; font-weight: 500; color: var(--sc-ink-2);
            margin-left: 0.22rem; text-decoration: none; }
.sc-val.sc-void { color: #46535e; }
.sc-note {
  font-family: 'Azeret Mono', monospace; font-size: 0.66rem; color: var(--sc-ink-3);
  margin-top: 0.42rem; letter-spacing: 0.02em;
}
.sc-row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; }
.sc-tag {
  display: inline-flex; align-items: center; gap: 0.42rem;
  font-family: 'Azeret Mono', monospace; font-size: 0.66rem; letter-spacing: 0.14em;
  text-transform: uppercase;
}
.sc-dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.sc-dot.sc-beat { animation: sc-beat 2s ease-in-out infinite; }
@keyframes sc-beat { 0%,100% { opacity: 1 } 50% { opacity: 0.25 } }
@media (prefers-reduced-motion: reduce) { .sc-dot.sc-beat { animation: none } }
.sc-mode {
  font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.85rem;
  letter-spacing: -0.02em; line-height: 1;
}
.sc-strip { display: flex; gap: 2px; margin-top: 0.75rem; }
.sc-strip div { flex: 1; height: 9px; border-radius: 1px; }
.sc-strip div.sc-nowcell { box-shadow: inset 0 0 0 1.5px rgba(255,255,255,0.85); }
/* The hero carries the same two facts as the plan band. Given the same geometry they read as
   a repeated template showing contradictory numbers, so here they are a footnote under a
   hairline and the room temperature keeps the viewport. */
.sc-hairline {
  margin-top: 0.75rem; padding-top: 0.7rem;
  border-top: 1px solid rgba(255,255,255,0.09);
}
.sc-strip.sc-thin { margin-top: 0.45rem; opacity: 0.72; }
.sc-strip.sc-thin div { height: 5px; }

/* ── sections below the fold ────────────────────────────────────────────── */
.sc-section { padding: 3.4rem 2.4rem 0; max-width: 1480px; margin: 0 auto; }
.sc-h2 {
  font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.6rem;
  letter-spacing: -0.025em; margin: 0 0 0.35rem;
}
.sc-sub { font-family: 'Azeret Mono', monospace; font-size: 0.72rem; color: var(--sc-ink-3);
          margin-bottom: 1.5rem; letter-spacing: 0.02em; }
.sc-rule { height: 1px; background: rgba(255,255,255,0.08); margin: 3.2rem 0 0; }
.sc-ledger { border-top: 1px solid rgba(255,255,255,0.1); }
.sc-ledger div {
  display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
  padding: 0.62rem 0.15rem; border-bottom: 1px solid rgba(255,255,255,0.07);
}
.sc-ledger span:first-child { font-size: 0.86rem; color: var(--sc-ink-2); }
.sc-ledger span:last-child {
  font-family: 'Azeret Mono', monospace; font-weight: 600; font-size: 0.94rem;
  font-variant-numeric: tabular-nums; color: var(--sc-ink);
}
.sc-tty {
  font-family: 'Azeret Mono', monospace; font-size: 0.7rem; color: #9fb0bd;
  background: #070b0e; border: 1px solid rgba(255,255,255,0.08); border-radius: 3px;
  padding: 0.8rem 0.95rem; overflow-x: auto; white-space: pre; line-height: 1.7;
}
@media (max-width: 1100px) {
  /* The orbit is pinned to a top lane and the content starts below it, so the orb has
     somewhere to be at every elevation instead of landing behind the dial. */
  .sc-orbit { top: 18px; height: 160px; }
  .sc-wrap { grid-template-columns: 1fr; justify-items: center; padding: 10.5rem 1.1rem 2.4rem; gap: 1.1rem; }
  .sc-stage { min-height: 100vh; }
  .sc-stage.sc-short { min-height: 72vh; }
  .sc-dial { max-width: 220px; }
  .sc-dialwrap { gap: 0.55rem; }
  .sc-nameplate { padding: 0.32rem 0.75rem; gap: 0.65rem; }
  .sc-np-read { font-size: 0.95rem; }
  .sc-val.sc-xl { font-size: 2.8rem; }
  .sc-val.sc-lg { font-size: 1.85rem; }
  .sc-plate { padding: 0.72rem 0.95rem; }
  .sc-plate.sc-lead { padding: 0.95rem 1.05rem 0.85rem; }
  .sc-stack { width: 100%; gap: 0.55rem; }
  .sc-row2 { gap: 0.55rem; }
  .sc-section { padding: 2.4rem 1.1rem 0; }
  .sc-skylabel { left: 1.1rem; bottom: 0.85rem; font-size: 0.62rem; }
}
@media (max-width: 640px) {
  /* Phone first viewport: clock + sky + temp + humidity must read without scrolling.
     Dew and the live strip may kiss the fold; they stay in the same stage. */
  .sc-orbit { top: 10px; height: 118px; }
  .sc-glow { width: 280px; height: 280px; margin: -140px 0 0 -140px; }
  .sc-orb { width: 42px; height: 42px; margin: -21px 0 0 -21px; }
  .sc-wrap { padding: 7.6rem 0.85rem 1.6rem; gap: 0.85rem; align-items: start; }
  .sc-stage { min-height: 100svh; }
  .sc-stage.sc-short { min-height: 70svh; }
  .sc-dial { max-width: 168px; }
  .sc-dialwrap { gap: 0.4rem; }
  .sc-nameplate { padding: 0.28rem 0.65rem; gap: 0.5rem; }
  .sc-np-cap { font-size: 0.56rem; letter-spacing: 0.14em; }
  .sc-np-read { font-size: 0.84rem; }
  .sc-val.sc-xl { font-size: 2.35rem; }
  .sc-val.sc-lg { font-size: 1.45rem; }
  .sc-val.sc-md { font-size: 1.15rem; }
  .sc-plate { padding: 0.55rem 0.75rem; box-shadow: 0 8px 18px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.07); }
  .sc-plate.sc-lead { padding: 0.7rem 0.85rem 0.6rem; }
  .sc-lbl { font-size: 0.58rem; letter-spacing: 0.14em; }
  .sc-note { font-size: 0.58rem; margin-top: 0.28rem; }
  .sc-stack { gap: 0.45rem; }
  .sc-row2 { gap: 0.45rem; }
  .sc-hairline { margin-top: 0.5rem; padding-top: 0.45rem; }
  .sc-mode { font-size: 1.35rem; }
  .sc-section { padding: 2rem 0.85rem 0; }
  .sc-h2 { font-size: 1.25rem; }
  .sc-skylabel { left: 0.85rem; right: 0.85rem; bottom: 0.55rem; font-size: 0.56rem; }
}
</style>
"""
