"""
dashboard/app.py
=================
Hamza's HEMS Dashboard.

Run from the project root:
    streamlit run dashboard/app.py

Depends on pipeline.py having been run at least once so that:
    logic/ac_predictor.pkl
    logic/peak_detector.pkl
    logic/peak_schedule.csv
all exist.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from logic.load_data import load_data
from logic.features import engineer_features
from logic.model import load_models
from logic.peak_detection import predict_peaks
from logic.control_logic import apply_control_logic, get_control_summary
from logic.scheduler import build_daily_schedule

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HEMS Smart A/C Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px; padding: 20px;
        border-left: 4px solid #00d4aa; margin-bottom: 10px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4aa; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .section-hdr {
        font-size: 1.2rem; font-weight: 600; color: #fff;
        margin: 18px 0 8px 0; border-bottom: 2px solid #00d4aa; padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load & run full pipeline (cached) ────────────────────────────────────────
@st.cache_data(show_spinner="Running pipeline…")
def load_pipeline():
    df_raw = load_data('HEMS_Sample_Dataset.xlsx')
    df     = engineer_features(df_raw)
    # Load pre-trained models (pipeline.py must have been run first)
    try:
        from logic.model import load_models
        load_models()   # just to verify they exist
    except Exception:
        from logic.model import train_ac_predictor, train_peak_detector
        train_ac_predictor(df)
        train_peak_detector(df)
    df = predict_peaks(df, lookahead_hours=2)
    df = apply_control_logic(df)
    return df

df   = load_pipeline()
kpis = get_control_summary(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏠 HEMS Control Panel")
st.sidebar.markdown("**GSR Hackathon 2026 — Energy Track**")
st.sidebar.markdown("---")

dates    = sorted(df['timestamp'].dt.date.astype(str).unique())
sel_date = st.sidebar.selectbox("📅 Select Day", dates)

buildings   = df['building_id'].unique().tolist()
sel_bldg    = st.sidebar.selectbox("🏠 Building", buildings)

st.sidebar.markdown("---")
st.sidebar.markdown("**Strategy**")
st.sidebar.markdown("🔵 **Pre-cool** — 2h before peak\n\n🔴 **Peak reduce** — 25% load cut\n\n🟠 **Comfort override** — dew point > 24°C")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏠 HEMS Smart A/C Optimization Dashboard")
st.markdown("Predictive demand response for residential buildings — Eastern Province, Saudi Arabia")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📊 Key Performance Indicators (30-Day)</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

def card(col, val, label):
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

card(c1, f"{kpis['peak_reduction_pct']}%",    "Peak Load Reduction")
card(c2, f"{kpis['energy_savings_pct']}%",    "Energy Savings")
card(c3, f"{kpis['comfort_pct']}%",           "Comfort Maintained")
card(c4, f"{kpis['total_saved_kwh']} kWh",    "Total kWh Saved")
card(c5, f"{kpis['peak_hours_reduced']} hrs",  "Peak Hours Reduced")

st.markdown("---")

# ── Daily filter ──────────────────────────────────────────────────────────────
day_df = df[
    (df['timestamp'].dt.date.astype(str) == sel_date) &
    (df['building_id'] == sel_bldg)
].sort_values('hour')

# ── Before vs After chart ─────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">⚡ Before vs After — Hourly A/C Load</div>', unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=day_df['hour'], y=day_df['ac_kwh'],
    name='Baseline (no optimization)',
    line=dict(color='#e04848', width=2.5, dash='dash'), mode='lines+markers'
))
fig.add_trace(go.Scatter(
    x=day_df['hour'], y=day_df['optimized_ac_kwh'],
    name='Optimized (HEMS control)',
    line=dict(color='#00d4aa', width=2.5), mode='lines+markers',
    fill='tonexty', fillcolor='rgba(0,212,170,0.07)'
))

# Shade peak hours red, pre-cool hours blue
for _, r in day_df[day_df['predicted_peak'] == 1].iterrows():
    fig.add_vrect(x0=r['hour']-.45, x1=r['hour']+.45,
                  fillcolor='rgba(224,72,72,0.13)', line_width=0)
for _, r in day_df[day_df['pre_cool_window'] == 1].iterrows():
    fig.add_vrect(x0=r['hour']-.45, x1=r['hour']+.45,
                  fillcolor='rgba(26,108,245,0.12)', line_width=0)

fig.update_layout(template='plotly_dark', height=370,
    xaxis_title='Hour of Day', yaxis_title='A/C Consumption (kWh)',
    legend=dict(orientation='h', y=1.08), margin=dict(l=40,r=20,t=10,b=40))
st.plotly_chart(fig, use_container_width=True)

# ── Row 2: comfort + mode pie ─────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown('<div class="section-hdr">😊 Comfort Score (Hourly)</div>', unsafe_allow_html=True)
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=day_df['hour'], y=day_df['comfort_score'],
        mode='lines+markers', line=dict(color='#f5a623', width=2.5)
    ))
    fig_c.add_hline(y=0.6, line_dash='dot', line_color='#888', annotation_text='min comfort threshold')
    fig_c.update_layout(template='plotly_dark', height=290,
        xaxis_title='Hour', yaxis_title='Comfort Score (0–1)',
        yaxis=dict(range=[0, 1.05]), margin=dict(l=40,r=20,t=10,b=40))
    st.plotly_chart(fig_c, use_container_width=True)

with col_r:
    st.markdown('<div class="section-hdr">🎛️ Control Mode Distribution</div>', unsafe_allow_html=True)
    mode_cnt = day_df['control_mode'].value_counts().reset_index()
    mode_cnt.columns = ['Mode', 'Hours']
    color_map = {'normal':'#3a7d44','pre_cool':'#1a6cf5',
                 'peak_reduce':'#e04848','comfort_override':'#c97d1b'}
    fig_p = px.pie(mode_cnt, values='Hours', names='Mode',
                   color='Mode', color_discrete_map=color_map, hole=0.45)
    fig_p.update_layout(template='plotly_dark', height=290, margin=dict(l=20,r=20,t=10,b=20))
    st.plotly_chart(fig_p, use_container_width=True)

# ── Climate strip ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🌡️ Climate — Temperature & Dew Point</div>', unsafe_allow_html=True)
fig_cl = go.Figure()
fig_cl.add_trace(go.Bar(x=day_df['hour'], y=day_df['temp'],
    name='Outdoor Temp (°C)', marker_color='#e04848', opacity=0.65))
fig_cl.add_trace(go.Scatter(x=day_df['hour'], y=day_df['dew_point'],
    name='Dew Point (°C)', mode='lines+markers',
    line=dict(color='#00d4aa', width=2), yaxis='y2'))
fig_cl.update_layout(template='plotly_dark', height=260,
    xaxis_title='Hour',
    yaxis=dict(title='Temperature (°C)'),
    yaxis2=dict(title='Dew Point (°C)', overlaying='y', side='right'),
    legend=dict(orientation='h', y=1.12),
    margin=dict(l=40,r=60,t=10,b=40))
st.plotly_chart(fig_cl, use_container_width=True)

# ── Hourly schedule table ─────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📅 Optimized Hourly Schedule</div>', unsafe_allow_html=True)

sched = build_daily_schedule(df, sel_date)
sched = sched[sched['hour'].isin(day_df[day_df['building_id'] == sel_bldg]['hour'])] if 'building_id' in day_df.columns else sched

def mode_color(val):
    return {
        'pre_cool':        'background-color:#1a3a6f; color:#7eb3ff',
        'peak_reduce':     'background-color:#6f1a1a; color:#ff8080',
        'comfort_override':'background-color:#6f4e1a; color:#ffc87a',
        'normal':          'background-color:#1a3a22; color:#7dff9a',
    }.get(val, '')

styled = sched.rename(columns={
    'hour':'Hour','temp':'Temp °C','humidity':'Humidity %','dew_point':'Dew Pt °C',
    'ac_kwh':'Baseline kWh','optimized_ac_kwh':'Optimized kWh','ac_saved_kwh':'Saved kWh',
    'control_mode':'Mode','setpoint_adj':'Setpoint Adj °C',
    'comfort_score':'Comfort','predicted_peak':'Peak','pre_cool_window':'Pre-cool',
    'action':'Action'
}).style.applymap(mode_color, subset=['Mode'])

st.dataframe(styled, use_container_width=True, height=420)

# ── 30-day savings bar chart ──────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📈 30-Day Daily Energy Savings</div>', unsafe_allow_html=True)

daily = df.groupby(df['timestamp'].dt.date.astype(str)).agg(
    Baseline=('ac_kwh', 'sum'),
    Optimized=('optimized_ac_kwh', 'sum')
).reset_index().rename(columns={'timestamp':'Date'})

fig_d = go.Figure()
fig_d.add_trace(go.Bar(x=daily['Date'], y=daily['Baseline'],
    name='Baseline', marker_color='#e04848', opacity=0.6))
fig_d.add_trace(go.Bar(x=daily['Date'], y=daily['Optimized'],
    name='Optimized', marker_color='#00d4aa', opacity=0.85))
fig_d.update_layout(template='plotly_dark', barmode='group', height=300,
    xaxis_title='Date', yaxis_title='Total A/C (kWh)',
    legend=dict(orientation='h', y=1.08),
    xaxis_tickangle=-45, margin=dict(l=40,r=20,t=10,b=70))
st.plotly_chart(fig_d, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("🏆 **GSR Hackathon 2026 — Energy Track** &nbsp;|&nbsp; HEMS Data-Driven Demand Response &nbsp;|&nbsp; Noor · Hana · Hamza · Rim · Zainab")