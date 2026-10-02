"""SmartCool console — an instrument dial under a live sky.

    streamlit run dashboard/app.py

Four sections, top to bottom:
  1. NOW         wall-clock dial, real sun over the site, and what the DHT11 is reading
  2. THE PLAN    the simulated demo day stepping hour by hour, with the mode it chose
  3. SAVINGS     what the load shift bought
  4. SENSORS     feed health, traces, session statistics, raw packets

Every sensor number is measured; indoor estimates are modelled; outdoor is replayed schedule.
The sky is not decoration: dashboard/skyclock.py puts the sun where it actually is over
config.SITE_LAT / SITE_LON, so the light on screen matches the light outside.
"""
import json
import os
import sys
import time
from datetime import date as _date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from dashboard.live_io import feed_status, load_history, merge_device, read_telemetry
from dashboard.skyclock import CSS, dial, sky, sky_layer
from logic.engine import get_payload

MODE_COLOR = {'normal': '#3f8f6b', 'pre_cool': '#4a86d8',
              'peak_reduce': '#d8a12a', 'comfort_override': '#cf5340'}
MODE_LABEL = {'normal': 'NORMAL', 'pre_cool': 'PRE-COOL',
              'peak_reduce': 'PEAK REDUCE', 'comfort_override': 'OVERRIDE'}
STATE_COLOR = {'live': '#5fd999', 'fault': '#ff8f6b', 'replay': '#c9a227',
               'stale': '#9a7a3a', 'offline': '#5d6d79'}
BRASS = '#c9a227'
INK_2, INK_3 = '#b6c4cf', '#9dabb6'
LOCAL_TZ = datetime.now().astimezone().tzinfo
# Trace colours are taken from the sky ramp (warm horizon, cool zenith, brass) so the charts
# stay inside the one accent the world committed to instead of inventing five hues.
C_WARM, C_COOL, C_PALE, C_MODEL, C_CLAY = '#e0a45c', '#7fa9c9', '#b9d4c9', '#6f9fe0', '#c2614a'

st.set_page_config(page_title='SmartCool Console', layout='wide',
                   initial_sidebar_state='collapsed')
st.markdown(CSS, unsafe_allow_html=True)
# Direction contract — kept in the emitted markup so it is auditable in the built page.
st.markdown("""<!--
THESIS: A thermostat's two facts are what time it is and how hot it is, so this console is an
instrument dial under a real sky, not a grid of metric cards.
OWN-WORLD: Aneroid weather-station instrument. Machined bezel, engraved tick ring, smoked
graphite plates, one brass accent. The page has no fixed palette: colour comes from solar
elevation over Dammam, from #03050b night to bleached Gulf haze at 82 degrees.
STORY: Read the room at a glance, watch the planned day step through, see what it saved,
then check the sensors are honest.
FIRST VIEWPORT: Full-bleed sky, sun on its true arc. 372px dial left, stacked readouts right
with room temperature dominant; the commanded mode and 24h strip ride under a hairline in the
feed plate, deliberately subordinate so nothing competes with the temperature.
FORM: Instrument dial, brief-pinned by the user (analog clock + time-of-day sky).
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review,
the verdict, and DESIGN.md
-->""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _demo():
    if not os.path.isfile(config.DEMO_DAY_PATH):
        return None
    with open(config.DEMO_DAY_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _kpis():
    p = os.path.join('results', 'kpis.json')
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _cached_payload(date, hour, scenario_name):
    scenario = config.SCENARIOS.get(scenario_name) if scenario_name else None
    return get_payload(date, int(hour), scenario=scenario)


def _num(v, nd=1, dash='—'):
    if v is None or (isinstance(v, float) and v != v):
        return dash
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return str(v)


def plate(label, value, unit='', note='', size='sc-lg', lead=False):
    void = ' sc-void' if value == '—' else ''
    u = f'<u>{unit}</u>' if unit else ''
    return (f"<div class='sc-plate{' sc-lead' if lead else ''}'><div class='sc-lbl'>{label}</div>"
            f"<div class='sc-val {size}{void}'>{value}{u}</div>"
            f"{f'<div class=sc-note>{note}</div>' if note else ''}</div>")


def figure(height=250, title='', legend=False):
    fig = go.Figure()
    top = 58 if (title and legend) else (34 if title else 12)
    fig.update_layout(
        height=height, margin=dict(l=52, r=16, t=top, b=34),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.016)',
        font=dict(family='Azeret Mono, monospace', color=INK_3, size=11),
        title=dict(text=title, font=dict(size=12, color=INK_2), y=0.97, yanchor='top'),
        showlegend=legend,
        legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0, font=dict(size=10),
                    bgcolor='rgba(0,0,0,0)'),
        hoverlabel=dict(bgcolor='#0d1217', bordercolor='rgba(255,255,255,0.14)',
                        font=dict(family='Azeret Mono, monospace', color='#eef3f7', size=11)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor='rgba(255,255,255,0.12)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False),
    )
    return fig


def chart(fig):
    """Plotly without the default modebar chrome — the console owns the surface."""
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False, 'responsive': True})


demo = _demo()
if demo is None:
    st.error('Run `python pipeline.py` first to generate results/demo_day.json')
    st.stop()

kpi = _kpis()
scenario_name = demo['scenario']['name'] if isinstance(demo.get('scenario'), dict) else None
rows = demo['rows']

if 'replay_hour' not in st.session_state:
    st.session_state.replay_hour = 0
    st.session_state.replay_started = time.time()
    st.session_state.replaying = False


def _on_replay_hour():
    st.session_state.replay_hour = st.session_state.replay_slider


def _live():
    """Re-read the feed. Called once per page run and again on every fragment tick."""
    tel = read_telemetry()
    hist = load_history()
    status = feed_status(tel, hist)
    return tel, hist, status, status['state'] in ('live', 'fault')


def _slot(tel, advance=False):
    """The simulated hour and its payload. The device feed owns the hour when it is
    publishing; otherwise the local replay walk does. Only the plan band advances it."""
    if tel and tel.get('fresh') and tel.get('sim_hour') is not None:
        sim_hour = int(tel['sim_hour'])
        sim_date = str(tel.get('date') or demo['date'])
    else:
        if advance and st.session_state.replaying:
            base = (config.SECONDS_PER_PEAK_HOUR if st.session_state.replay_hour in config.PEAK_HOURS
                    else config.SECONDS_PER_SIM_HOUR)
            if time.time() - st.session_state.replay_started >= base:
                st.session_state.replay_hour = (st.session_state.replay_hour + 1) % 24
                st.session_state.replay_started = time.time()
        sim_hour, sim_date = st.session_state.replay_hour, demo['date']
    payload = merge_device(_cached_payload(sim_date, sim_hour, scenario_name), tel)
    return sim_hour, sim_date, payload


def _strip(payload):
    return ''.join(
        f"<div class='{'sc-nowcell' if r['now'] else ''}' "
        f"style='background:{MODE_COLOR.get(r['actual_mode'], '#444')};"
        f"opacity:{1 if r['now'] else 0.5}'></div>" for r in payload['ribbon']
    )


tel, hist, status, device_live = _live()
sim_hour, sim_date, payload = _slot(tel)
mode = payload['modelled']['actual_mode']['value']
mode_c = MODE_COLOR.get(mode, '#889')
# The hour can only move when the board is publishing or the replay walk is running; outside
# those two cases the bands are static and nothing needs to tick.
TICKING = device_live or st.session_state.replaying

with st.sidebar:
    st.markdown("<div class='sc-lbl'>SmartCool · GSR 2026</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sc-tag' style='color:{STATE_COLOR[status['state']]};margin:.55rem 0 1.1rem'>"
                f"<span class='sc-dot{' sc-beat' if device_live else ''}' "
                f"style='background:{STATE_COLOR[status['state']]}'></span>{status['badge']}</div>",
                unsafe_allow_html=True)
    st.caption('Service panel · connect the board, then start the bridge before powering the ESP32:')
    st.code('python bridge/serial_bridge.py --port /dev/cu.usbserial-0001', language=None)
    st.caption('The laptop paces the simulated hour. The ESP32 reads the DHT about once a '
               'second and streams every sample.')
    st.markdown('---')
    if st.button('Stop replay' if st.session_state.replaying else 'Replay the planned day',
                 width='stretch', disabled=device_live):
        st.session_state.replaying = not st.session_state.replaying
        st.session_state.replay_started = time.time()
        st.rerun()
    st.caption('Replay walks the schedule so the plan section moves without hardware. '
               'Measured tiles stay empty — nothing is sensing yet.')
    st.session_state.replay_slider = st.session_state.replay_hour
    st.slider('Simulated hour', 0, 23, key='replay_slider', on_change=_on_replay_hour,
              disabled=device_live or st.session_state.replaying)

# ══ 1. NOW ═══════════════════════════════════════════════════════════════════
# Only the hero ticks. A page-wide rerun every second rebuilt the savings charts and
# collapsed the decision trace, so the clock's cadence is scoped to the band that needs it.
@st.fragment(run_every=1)
def now_band():
    tel, _hist, status, device_live = _live()
    _h, _d, pl = _slot(tel)
    md = pl['modelled']['actual_mode']['value']
    md_c = MODE_COLOR.get(md, '#889')
    dev = pl['device']

    now = datetime.now()
    secs = now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6

    t_live, rh_live = dev['temp']['value'], dev['rh']['value']
    dew_live, fan_live = dev['dew']['value'], dev['fan_duty']['value']
    measured, waiting = 'DHT11 · measured', 'waiting for the board'
    feed_note = (f"{status['rate_hz'] or '—'} Hz · last packet {status['age_s']:.1f}s ago"
                 if device_live else 'no packets yet — start the bridge')
    fan = fan_live if fan_live is not None else pl['modelled']['fan_pwm']['value']

    readouts = (
        plate('Room temperature', _num(t_live), '°C' if t_live is not None else '',
              measured if t_live is not None else waiting, 'sc-xl', lead=True)
        + "<div class='sc-row2'>"
        + plate('Relative humidity', _num(rh_live, 0), '%' if rh_live is not None else '',
                measured if rh_live is not None else waiting, 'sc-lg')
        + plate('Dew point', _num(dew_live), '°C' if dew_live is not None else '',
                (f"{measured} · muggy above {config.DEW_POINT_UNCOMFORTABLE:g} °C"
                 if dew_live is not None else waiting),
                'sc-lg')
        + "</div>"
        + f"<div class='sc-plate'><div class='sc-tag' style='color:{STATE_COLOR[status['state']]}'>"
          f"<span class='sc-dot{' sc-beat' if device_live else ''}' "
          f"style='background:{STATE_COLOR[status['state']]}'></span>{status['badge']}</div>"
          f"<div class='sc-note' style='margin-top:.45rem'>{feed_note} · "
          f"{status['samples']} samples · fan {_num(fan, 0)}%</div>"
          f"<div class='sc-tag sc-hairline' style='color:{md_c}'>"
          f"commanded now · {MODE_LABEL.get(md, md)}</div>"
          f"<div class='sc-strip sc-thin'>{_strip(pl)}</div></div>"
    )
    st.markdown(
        f"<div class='sc-stage'>{sky_layer(sky(now.timetuple().tm_yday, secs / 3600.0))}"
        f"<div class='sc-wrap'>"
        f"<div>{dial('live', seconds_into_day=secs, caption=now.strftime('%a %d %b'), readout=now.strftime('%H:%M'), accent=BRASS)}</div>"
        f"<div class='sc-stack'>{readouts}</div></div></div>",
        unsafe_allow_html=True,
    )


now_band()


# ══ 2. THE PLAN ══════════════════════════════════════════════════════════════
@st.fragment(run_every=1 if TICKING else None)
def plan_band():
    tel, _hist, _status, _live_dev = _live()
    s_hour, s_date, pl = _slot(tel, advance=True)
    md = pl['modelled']['actual_mode']['value']
    md_c = MODE_COLOR.get(md, '#889')
    sim_sky = sky(_date.fromisoformat(s_date).timetuple().tm_yday, s_hour + 0.5)
    plan_plates = (
        f"<div class='sc-plate sc-lead'><div class='sc-lbl'>Mode · hour {s_hour:02d}:00</div>"
        f"<div class='sc-mode' style='color:{md_c};margin-top:.45rem'>{MODE_LABEL.get(md, md)}</div>"
        f"<div class='sc-note'>{pl['modelled']['reason']['value']}</div>"
        f"<div class='sc-strip'>{_strip(pl)}</div></div>"
        + "<div class='sc-row2'>"
        + plate('Outdoor', _num(pl['replayed']['outdoor_temp']['value']), '°C',
                f"dew {_num(pl['replayed']['outdoor_dew']['value'])} °C · replayed", 'sc-lg')
        + plate('Indoor model', _num(pl['modelled']['indoor_temp_est_c']['value']), '°C',
                f"comfort cap {config.COMFORT_T_MAX:g} °C · modelled", 'sc-lg')
        + "</div>"
        + "<div class='sc-row2'>"
        + plate('Saved so far', _num(pl['counters']['kwh_saved_net']['value'], 2), 'kWh',
                'modelled, cumulative to this hour', 'sc-md')
        + plate('Peak kWh cut', _num(pl['counters']['peak_kwh_cut']['value'], 2), 'kWh',
                f"tariff peak {config.PEAK_HOURS[0]:02d}:00–{config.PEAK_HOURS[-1]:02d}:59", 'sc-md')
        + "</div>"
    )
    st.markdown(
        f"<div class='sc-stage sc-short'>{sky_layer(sim_sky, 'simulated day · ' + s_date)}"
        f"<div class='sc-wrap'>"
        f"<div>{dial('step', hour=s_hour, caption=scenario_name or 'planned day', readout=f'{s_hour:02d}:00', accent=md_c)}</div>"
        f"<div class='sc-stack'>{plan_plates}</div></div></div>",
        unsafe_allow_html=True,
    )


plan_band()

# ══ 3. SAVINGS ═══════════════════════════════════════════════════════════════
st.markdown("<div class='sc-section'><div class='sc-h2'>What the shift bought</div>"
            "<div class='sc-sub'>The demo day hour by hour, then the 29-day simulation totals. "
            "Indoor temperature is modelled; the dataset is a synthetic sample.</div></div>",
            unsafe_allow_html=True)

sec = st.container()
with sec:
    pad_l, body, pad_r = st.columns([0.055, 0.89, 0.055])
    with body:
        c1, c2 = st.columns([1.55, 1])
        with c1:
            hours = [r['hour'] for r in rows]
            base = [r['kwh_baseline'] for r in rows]
            opt = [r['kwh_optimized'] for r in rows]
            fig = figure(312, 'A/C load across the demo day (kWh per hour)', legend=True)
            fig.add_vrect(x0=config.PEAK_HOURS[0] - 0.5, x1=config.PEAK_HOURS[-1] + 0.5,
                          fillcolor='rgba(216,161,42,0.10)', line_width=0,
                          annotation_text='tariff peak', annotation_position='top left',
                          annotation_font=dict(size=10, color=BRASS))
            fig.add_trace(go.Scatter(x=hours, y=base, name='baseline', mode='lines',
                                     line=dict(color='#7d8b97', width=1.6, dash='dot')))
            fig.add_trace(go.Scatter(x=hours, y=opt, name='SmartCool', mode='lines',
                                     line=dict(color=BRASS, width=2.6), fill='tozeroy',
                                     fillcolor='rgba(201,162,39,0.14)'))
            fig.update_xaxes(dtick=3, title=None)
            chart(fig)

            cum, run = [], 0.0
            for r in rows:
                run += r['kwh_baseline'] - r['kwh_optimized']
                cum.append(run)
            fig2 = figure(206, 'Cumulative kWh saved (modelled)')
            fig2.add_trace(go.Scatter(x=hours, y=cum, mode='lines', line=dict(color=C_COOL, width=2.4),
                                      fill='tozeroy', fillcolor='rgba(127,169,201,0.13)'))
            fig2.add_hline(y=0, line=dict(color='rgba(255,255,255,0.22)', width=1))
            fig2.update_xaxes(dtick=3)
            chart(fig2)
        with c2:
            cost = kpi.get('cost_sar', {})
            cond = kpi.get('condensate_L_day', {})
            pay = kpi.get('payback_months', {})
            led = [
                ('Peak demand reduction', f"{kpi.get('peak_reduction_pct', 0):.1f} %"),
                ('Net energy change', f"{kpi.get('energy_change_pct', 0):+.2f} %"),
                ('Energy, open loop', f"{kpi.get('energy_change_open_loop_pct', 0):+.2f} %"),
                ('Bill saving · time-of-use', f"{cost.get('saved_per_year_tou', 0):.0f} SAR/yr"),
                ('Bill saving · flat tariff', f"{cost.get('saved_per_year_flat', 0):.0f} SAR/yr"),
                ('CO₂ avoided', f"{kpi.get('co2_kg', {}).get('saved_per_year', 0):.0f} kg/yr"),
                ('Condensate recovered', f"{cond.get('optimized', 0):.1f} L/day"),
                ('Hours above comfort cap',
                 f"{kpi.get('hours_above_limit', 0)} of {kpi.get('scope', {}).get('hours', 0)}"),
                ('Warmest modelled indoor', f"{kpi.get('max_indoor_c', 0):.2f} °C"),
                ('Payback · time-of-use', f"{pay.get('tou', 0):.1f} months"),
            ]
            st.markdown(
                "<div class='sc-lbl' style='margin-bottom:.6rem'>29-day simulation</div>"
                "<div class='sc-ledger'>"
                + ''.join(f"<div><span>{k}</span><span>{v}</span></div>" for k, v in led)
                + "</div>"
                + f"<div class='sc-note' style='margin-top:.9rem'>Flat tariff today is "
                  f"{config.TARIFF_FLAT_SAR} SAR/kWh, so the shift itself earns nothing there — "
                  f"only the net kWh does.</div>",
                unsafe_allow_html=True,
            )

# ══ 4. SENSORS ═══════════════════════════════════════════════════════════════
st.markdown("<div class='sc-section'><div class='sc-rule'></div></div>", unsafe_allow_html=True)
st.markdown("<div class='sc-section'><div class='sc-h2'>Sensors</div>"
            "<div class='sc-sub'>Everything the board has reported this session. "
            "Empty until the DHT11 streams.</div></div>", unsafe_allow_html=True)

# Five seconds, not one: the traces grow slowly enough that a slower tick costs nothing and
# leaves the charts usable between redraws.
@st.fragment(run_every=5 if TICKING else None)
def sensor_band():
    _tel, hist, status, _live_dev = _live()
    pad_l, body, pad_r = st.columns([0.055, 0.89, 0.055])
    with body:
        if not hist:
            st.markdown(
                "<div class='sc-plate'><div class='sc-lbl'>No samples recorded</div>"
                "<div class='sc-note' style='margin-top:.5rem'>Traces, session statistics and the "
                "packet tail draw themselves as the feed arrives. Start the bridge, or use "
                "<code>python tools/fake_device_feed.py</code> to exercise the path without a board."
                "</div></div>", unsafe_allow_html=True)
        else:
            hdf = pd.DataFrame(hist)
            # to_datetime(unit='s') lands in UTC. The packet tail below stamps with
            # fromtimestamp(), which is local, so leaving this naive put the traces three
            # hours off the packets in the one section whose job is proving the feed honest.
            hdf['t'] = (pd.to_datetime(hdf.get('updated_unix'), unit='s', errors='coerce', utc=True)
                        .dt.tz_convert(LOCAL_TZ).dt.tz_localize(None))
            hdf = hdf.dropna(subset=['t']).sort_values('t')
            has = [c for c in ('temp', 'rh', 'dew') if c in hdf.columns]
            srows = hdf[hdf[has].notna().any(axis=1)] if has else hdf.iloc[0:0]
            pdf = srows if len(srows) else hdf

            s1, s2, s3 = st.columns(3)
            for col, (key, label, colr, extra) in zip(
                (s1, s2, s3),
                (('temp', 'Room temperature (°C)', C_WARM,
                  ('indoor_temp_est_c', 'modelled indoor', C_MODEL)),
                 ('rh', 'Relative humidity (%) · measured', C_COOL, None),
                 ('dew', 'Dew point (°C) · measured', C_PALE, None)),
            ):
                with col:
                    has_extra = bool(extra and extra[0] in pdf.columns and pdf[extra[0]].notna().any())
                    f = figure(196, label, legend=has_extra)
                    if key in pdf.columns and pdf[key].notna().any():
                        f.add_trace(go.Scatter(x=pdf['t'], y=pdf[key], mode='lines',
                                               line=dict(color=colr, width=2),
                                               name='measured'))
                    if has_extra:
                        f.add_trace(go.Scatter(x=pdf['t'], y=pdf[extra[0]], mode='lines',
                                               name=extra[1],
                                               line=dict(color=extra[2], width=1.3, dash='dot')))
                    chart(f)

            s4, s5 = st.columns([1.7, 1])
            with s4:
                f = figure(206, 'Fan duty (%) against outdoor temperature', legend=True)
                if 'fan_pwm' in hdf.columns:
                    f.add_trace(go.Scatter(x=hdf['t'], y=hdf['fan_pwm'], mode='lines',
                                           name='fan · measured',
                                           line=dict(color=BRASS, width=2), fill='tozeroy',
                                           fillcolor='rgba(201,162,39,0.10)'))
                if 'outdoor_temp' in hdf.columns:
                    f.add_trace(go.Scatter(x=hdf['t'], y=hdf['outdoor_temp'], mode='lines',
                                           name='outdoor · replayed', yaxis='y2',
                                           line=dict(color=C_CLAY, width=1.4, dash='dot')))
                    f.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False,
                                                tickfont=dict(color=C_CLAY)))
                chart(f)
            with s5:
                stat = []
                for key, label, unit in (('temp', 'Room temp', '°C'), ('rh', 'Humidity', '%'),
                                         ('dew', 'Dew point', '°C'), ('fan_pwm', 'Fan duty', '%')):
                    if key in hdf.columns and hdf[key].notna().any():
                        s = pd.to_numeric(hdf[key], errors='coerce').dropna()
                        stat.append((label, f'{s.min():g} / {s.mean():.1f} / {s.max():g} {unit}'))
                st.markdown(
                    "<div class='sc-lbl' style='margin-bottom:.6rem'>Session min / mean / max · measured</div>"
                    "<div class='sc-ledger'>"
                    + ''.join(f"<div><span>{k}</span><span>{v}</span></div>" for k, v in stat)
                    + f"<div><span>Samples</span><span>{status['samples']}</span></div></div>",
                    unsafe_allow_html=True)

            lines = []
            for r in hist[-14:][::-1]:
                ts = datetime.fromtimestamp(float(r.get('updated_unix') or 0)).strftime('%H:%M:%S')
                lines.append(
                    f"{ts}  {(r.get('source') or '?')[:5]:<5}  h{int(r.get('sim_hour') or 0):02d}  "
                    f"{str(r.get('actual_mode') or '-'):<16} T={_num(r.get('temp'))}  "
                    f"RH={_num(r.get('rh'), 0)}  dew={_num(r.get('dew'))}  "
                    f"fan={_num(r.get('fan_pwm'), 0)}  ovr={r.get('override_source') or '-'}")
            st.markdown("<div class='sc-lbl' style='margin:1rem 0 .5rem'>Raw packets</div>"
                        f"<div class='sc-tty'>{'<br>'.join(lines)}</div>", unsafe_allow_html=True)


sensor_band()

# Outside the ticking fragments so opening it survives every redraw above.
with st.container():
    pad_l, body, pad_r = st.columns([0.055, 0.89, 0.055])
    with body:
        with st.expander(f'Why hour {sim_hour:02d}:00 chose {MODE_LABEL.get(mode, mode)}'):
            for line in payload.get('decision_trace', []):
                st.text(line)
