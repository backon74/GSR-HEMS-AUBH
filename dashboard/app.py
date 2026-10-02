"""SmartCool sensor console.

    streamlit run dashboard/app.py

A live view of what the ESP32 + DHT11 are reporting: feed health, current readings, streaming
traces against wall-clock time, the raw packet tail, and session statistics that accumulate.
The laptop paces the simulated hour (config.SECONDS_PER_SIM_HOUR); the device only senses and
executes, so every sensor number here is measured, not modelled.
"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from dashboard.live_io import feed_status, load_history, merge_device, read_telemetry
from logic.engine import get_payload

MODE_COLOR = {
    'normal': '#3d8b6e',
    'pre_cool': '#2f6fed',
    'peak_reduce': '#e0a106',
    'comfort_override': '#d64545',
}
MODE_LABEL = {
    'normal': 'NORMAL',
    'pre_cool': 'PRE-COOL',
    'peak_reduce': 'PEAK REDUCE',
    'comfort_override': 'OVERRIDE',
}
STATE_COLOR = {'live': '#7dffb3', 'fault': '#ff8f6b', 'replay': '#f0c674',
               'stale': '#b58b4c', 'offline': '#6f8593'}

st.set_page_config(page_title='SmartCool Sensor Console', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Sora:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #0a0f12; color: #e8eef2; }
.block-container { padding-top: 1rem; max-width: 1500px; }
.sc-bar {
  display: flex; align-items: center; gap: 1.4rem; flex-wrap: wrap;
  border: 1px solid rgba(232,238,242,0.1); border-left: 4px solid #6f8593;
  background: rgba(255,255,255,0.025); padding: 0.75rem 1rem; margin-bottom: 1rem;
}
.sc-bar .pill { font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; font-weight: 600; letter-spacing: 0.06em; }
.sc-bar .kv { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #8aa0ad; }
.sc-bar .kv b { color: #d7e2e9; font-weight: 500; }
.sc-title { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.015em; margin-right: auto; }
.sc-gauge {
  border: 1px solid rgba(232,238,242,0.09); background: rgba(255,255,255,0.03);
  padding: 0.85rem 1rem 0.95rem;
}
.sc-gauge .lbl { color: #8aa0ad; font-size: 0.74rem; letter-spacing: 0.04em; }
.sc-gauge .big { font-family: 'IBM Plex Mono', monospace; font-size: 2.5rem; font-weight: 600; line-height: 1.1; }
.sc-gauge .unit { font-size: 1rem; color: #8aa0ad; margin-left: 0.15rem; }
.sc-gauge .sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #6f8593; margin-top: 0.3rem; }
.sc-mode { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600;
           padding: 0.9rem 1rem; border-left: 5px solid #3d8b6e; background: rgba(255,255,255,0.03); }
.sc-feed { font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #9fb3c0;
           background: #070b0d; border: 1px solid rgba(232,238,242,0.08); padding: 0.6rem 0.8rem;
           max-height: 190px; overflow-y: auto; white-space: pre; line-height: 1.5; }
.sc-strip { display: flex; gap: 2px; margin: 0.2rem 0 0.9rem; }
.sc-strip div { flex: 1; height: 10px; }
h3 { font-family: 'Sora', sans-serif !important; font-size: 1.05rem !important; letter-spacing: -0.01em;
     margin-top: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _demo():
    if not os.path.isfile(config.DEMO_DAY_PATH):
        return None
    with open(config.DEMO_DAY_PATH) as f:
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


def gauge(label, value, unit, sub, accent='#e8eef2'):
    st.markdown(
        f'<div class="sc-gauge"><div class="lbl">{label}</div>'
        f'<div class="big" style="color:{accent}">{value}<span class="unit">{unit}</span></div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def spark(df, ycol, color, title, unit, height=170, extra=None):
    fig = go.Figure()
    if ycol in df.columns and df[ycol].notna().any():
        fig.add_trace(go.Scatter(x=df['t'], y=df[ycol], mode='lines',
                                 line=dict(color=color, width=2), name=unit,
                                 fill='tozeroy', fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba')
                                 if color.startswith('rgb') else 'rgba(255,255,255,0.04)'))
    for name, col, c in (extra or []):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df['t'], y=df[col], mode='lines', name=name,
                                     line=dict(color=c, width=1.4, dash='dot')))
    fig.update_layout(height=height, margin=dict(l=44, r=12, t=28, b=28),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.015)',
                      font_color='#9fb3c0', font_size=11, title=dict(text=title, font_size=12),
                      showlegend=bool(extra), legend=dict(orientation='h', y=1.3, font_size=10),
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.06)'))
    st.plotly_chart(fig, width='stretch')


demo = _demo()
if demo is None:
    st.error('Run `python pipeline.py` first to generate results/demo_day.json')
    st.stop()

scenario_name = demo['scenario']['name'] if isinstance(demo.get('scenario'), dict) else None

if 'replay_hour' not in st.session_state:
    st.session_state.replay_hour = 0
    st.session_state.replay_started = time.time()
    st.session_state.replaying = False


def _on_replay_hour():
    st.session_state.replay_hour = st.session_state.replay_slider

tel = read_telemetry()
hist = load_history()
status = feed_status(tel, hist)
device_live = status['state'] in ('live', 'fault')

# Hour context: from the feed when it is publishing, else the local replay walk
if tel and tel.get('fresh') and tel.get('sim_hour') is not None:
    hour = int(tel['sim_hour'])
    date = str(tel.get('date') or demo['date'])
elif st.session_state.replaying:
    base = (config.SECONDS_PER_PEAK_HOUR if st.session_state.replay_hour in config.PEAK_HOURS
            else config.SECONDS_PER_SIM_HOUR)
    if time.time() - st.session_state.replay_started >= base:
        st.session_state.replay_hour = (st.session_state.replay_hour + 1) % 24
        st.session_state.replay_started = time.time()
    hour, date = st.session_state.replay_hour, demo['date']
else:
    hour, date = st.session_state.replay_hour, demo['date']

with st.sidebar:
    st.markdown('### Sensor console')
    st.caption('SmartCool · GSR 2026 Energy')
    st.markdown('**Feed**')
    st.markdown(
        f"<span class='pill' style='color:{STATE_COLOR[status['state']]}'>{status['badge']}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        'Connect the board to populate the sensor panels:\n\n'
        '`python bridge/serial_bridge.py --port /dev/cu.usbserial-0001`\n\n'
        'Start the bridge before powering the ESP32. The laptop paces the hour; '
        'the device reads the DHT about once a second and streams every sample.'
    )
    st.code(config.LIVE_TELEMETRY_PATH, language=None)
    st.markdown('---')
    st.markdown('**No board connected?**')
    if st.button('Stop replay' if st.session_state.replaying else 'Replay schedule',
                 width='stretch', disabled=device_live):
        st.session_state.replaying = not st.session_state.replaying
        st.session_state.replay_started = time.time()
        st.rerun()
    st.caption('Replay walks the planned day so the layout is reviewable. Sensor tiles stay '
               'empty because nothing is measuring yet.')
    # Mirror into a separate widget key; a disabled slider keyed on replay_hour would
    # overwrite the auto-advanced hour on every rerun.
    st.session_state.replay_slider = st.session_state.replay_hour
    st.slider('Replay hour', 0, 23, key='replay_slider', on_change=_on_replay_hour,
              disabled=device_live or st.session_state.replaying)

payload = merge_device(_cached_payload(date, hour, scenario_name), tel)
mode = payload['modelled']['actual_mode']['value']
mode_c = MODE_COLOR.get(mode, '#888')
dev = payload['device']

age = status['age_s']
age_txt = '—' if age is None else (f'{age:.1f}s ago' if age < 600 else 'stale')
rate_txt = '—' if not status['rate_hz'] else f"{status['rate_hz']} Hz"
ok = status['sensor_ok']
ok_txt = 'ok' if ok is True else ('FAULT' if ok is False else '—')

st.markdown(
    f"<div class='sc-bar' style='border-left-color:{STATE_COLOR[status['state']]}'>"
    f"<div class='sc-title'>SmartCool sensor console</div>"
    f"<div class='pill' style='color:{STATE_COLOR[status['state']]}'>{status['badge']}</div>"
    f"<div class='kv'>last packet <b>{age_txt}</b></div>"
    f"<div class='kv'>sample rate <b>{rate_txt}</b></div>"
    f"<div class='kv'>DHT <b>{ok_txt}</b></div>"
    f"<div class='kv'>samples <b>{status['samples']}</b></div>"
    f"<div class='kv'>override <b>{(tel or {}).get('override_source', '—')}</b></div>"
    f"<div class='kv'>{date} · hour <b>{hour:02d}:00</b></div>"
    f"</div>",
    unsafe_allow_html=True,
)

if not device_live:
    st.info('Waiting for the ESP32 DHT stream. Sensor tiles and traces below fill as packets arrive; '
            'schedule context (outdoor, modelled indoor, plan) is shown meanwhile.')

# ── Measured now ─────────────────────────────────────────────────────────────
st.markdown('### Measured now')
g = st.columns(4)
with g[0]:
    gauge('Room temperature', _num(dev['temp']['value']), '°C',
          'DHT11 · measured' if dev['temp']['value'] is not None else 'no sample yet',
          '#7dffb3' if dev['temp']['value'] is not None else '#44525c')
with g[1]:
    gauge('Relative humidity', _num(dev['rh']['value'], 0), '%',
          'DHT11 · measured' if dev['rh']['value'] is not None else 'no sample yet',
          '#5fd0f3' if dev['rh']['value'] is not None else '#44525c')
with g[2]:
    gauge('Dew point', _num(dev['dew']['value']), '°C',
          f"override at ≥ {config.DEW_POINT_UNCOMFORTABLE:g} °C" if dev['dew']['value'] is not None
          else 'no sample yet',
          '#a7f3d0' if dev['dew']['value'] is not None else '#44525c')
with g[3]:
    fan = dev['fan_duty']['value'] if dev['fan_duty']['value'] is not None else payload['modelled']['fan_pwm']['value']
    gauge('Fan duty', _num(fan, 0), '%',
          'device reported' if dev['fan_duty']['value'] is not None else 'commanded',
          '#f0c674')

st.markdown(
    f"<div class='sc-mode' style='border-left-color:{mode_c};color:{mode_c}'>{MODE_LABEL.get(mode, mode)}"
    f"<span style='font-size:0.8rem;color:#8aa0ad;margin-left:0.8rem'>"
    f"{payload['modelled']['reason']['value']}</span></div>",
    unsafe_allow_html=True,
)
strip = ''.join(
    f"<div style='background:{MODE_COLOR.get(r['actual_mode'], '#444')};"
    f"opacity:{1 if r['now'] else 0.45}'></div>" for r in payload['ribbon']
)
st.markdown(f"<div class='sc-strip'>{strip}</div>", unsafe_allow_html=True)

# ── Context the sensors are being judged against ─────────────────────────────
c = st.columns(4)
with c[0]:
    gauge('Outdoor temp', _num(payload['replayed']['outdoor_temp']['value']), '°C', 'replayed schedule', '#e07a5f')
with c[1]:
    gauge('Outdoor dew', _num(payload['replayed']['outdoor_dew']['value']), '°C', 'replayed schedule', '#c98f6b')
with c[2]:
    gauge('Indoor model', _num(payload['modelled']['indoor_temp_est_c']['value']), '°C',
          f"comfort cap {config.COMFORT_T_MAX:g} °C", '#7eb6ff')
with c[3]:
    gauge('kWh saved today', _num(payload['counters']['kwh_saved_net']['value'], 2), '',
          'modelled cumulative', '#9fb3c0')

# ── Live traces ──────────────────────────────────────────────────────────────
st.markdown('### Live traces')
if not hist:
    st.caption('No packets recorded yet — traces draw themselves as the feed arrives.')
else:
    hdf = pd.DataFrame(hist)
    hdf['t'] = pd.to_datetime(hdf.get('updated_unix'), unit='s', errors='coerce')
    hdf = hdf.dropna(subset=['t']).sort_values('t')
    sensor_rows = hdf[hdf[['temp', 'rh', 'dew']].notna().any(axis=1)] if 'temp' in hdf.columns else hdf.iloc[0:0]
    plot_df = sensor_rows if len(sensor_rows) else hdf

    t1, t2, t3 = st.columns(3)
    with t1:
        spark(plot_df, 'temp', '#7dffb3', 'Room temperature (°C)', '°C',
              extra=[('indoor model', 'indoor_temp_est_c', '#7eb6ff')])
    with t2:
        spark(plot_df, 'rh', '#5fd0f3', 'Relative humidity (%)', '%')
    with t3:
        spark(plot_df, 'dew', '#a7f3d0', 'Dew point (°C)', '°C')

    t4, t5 = st.columns([2, 1])
    with t4:
        spark(hdf, 'fan_pwm', '#f0c674', 'Fan duty (%) with outdoor temperature', '%',
              extra=[('outdoor °C', 'outdoor_temp', '#e07a5f')])
    with t5:
        if 'actual_mode' in hdf.columns:
            counts = hdf['actual_mode'].value_counts()
            fig = go.Figure(go.Bar(
                x=counts.values, y=[MODE_LABEL.get(m, m) for m in counts.index], orientation='h',
                marker_color=[MODE_COLOR.get(m, '#555') for m in counts.index]))
            fig.update_layout(height=170, margin=dict(l=90, r=12, t=28, b=28),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.015)',
                              font_color='#9fb3c0', font_size=11,
                              title=dict(text='Packets per mode', font_size=12),
                              xaxis=dict(gridcolor='rgba(255,255,255,0.06)'), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, width='stretch')

    # ── Session statistics ───────────────────────────────────────────────────
    st.markdown('### Session statistics')
    rows = []
    for key, label, unit in (('temp', 'Room temperature', '°C'), ('rh', 'Relative humidity', '%'),
                             ('dew', 'Dew point', '°C'), ('fan_pwm', 'Fan duty', '%'),
                             ('outdoor_temp', 'Outdoor temperature', '°C'),
                             ('indoor_temp_est_c', 'Indoor model', '°C')):
        if key in hdf.columns and hdf[key].notna().any():
            s = pd.to_numeric(hdf[key], errors='coerce').dropna()
            rows.append({'signal': label, 'unit': unit, 'samples': int(s.size),
                         'min': round(s.min(), 2), 'mean': round(s.mean(), 2),
                         'max': round(s.max(), 2), 'latest': round(s.iloc[-1], 2)})
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.markdown('### Raw packet tail')
    lines = []
    for r in hist[-14:][::-1]:
        ts = datetime.fromtimestamp(float(r.get('updated_unix') or 0)).strftime('%H:%M:%S')
        src = (r.get('source') or '?')[:5]
        lines.append(
            f"{ts}  {src:<5}  h{int(r.get('sim_hour') or 0):02d}  {str(r.get('actual_mode') or '-'):<16} "
            f"T={_num(r.get('temp'))}  RH={_num(r.get('rh'), 0)}  Dew={_num(r.get('dew'))} "
            f"fan={_num(r.get('fan_pwm'), 0)}  ovr={r.get('override_source') or '-'}"
        )
    st.markdown(f"<div class='sc-feed'>{'<br>'.join(lines)}</div>", unsafe_allow_html=True)

with st.expander('Decision trace for this hour'):
    for line in payload.get('decision_trace', []):
        st.text(line)

if device_live or st.session_state.replaying:
    time.sleep(1.0)
    st.rerun()
