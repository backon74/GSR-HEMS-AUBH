"""
HEMS Dashboard.

Run from the project root:
    streamlit run dashboard/app.py
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
from logic.condensate import add_condensate_columns, get_condensate_summary
from logic.cost_savings import add_cost_columns, get_cost_summary, get_scaling_summary

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
    .metric-card-blue  { border-left-color: #1a6cf5 !important; }
    .metric-card-amber { border-left-color: #f5a623 !important; }
    .metric-card-water { border-left-color: #38bdf8 !important; }
    .metric-card-red   { border-left-color: #e04848 !important; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4aa; }
    .metric-value-blue  { color: #1a6cf5 !important; }
    .metric-value-amber { color: #f5a623 !important; }
    .metric-value-water { color: #38bdf8 !important; }
    .metric-value-red   { color: #e04848 !important; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .section-hdr {
        font-size: 1.2rem; font-weight: 600; color: #fff;
        margin: 18px 0 8px 0; border-bottom: 2px solid #00d4aa; padding-bottom: 5px;
    }
    .scale-box {
        background: linear-gradient(135deg, #1a2240, #2a1a40);
        border-radius: 12px; padding: 18px;
        border: 1px solid #3a3a6a; margin-bottom: 8px;
    }
    .scale-title { font-size: 0.8rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .scale-val   { font-size: 1.6rem; font-weight: 700; color: #c084fc; margin: 4px 0; }
    .scale-sub   { font-size: 0.75rem; color: #888; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Running pipeline…")
def load_pipeline():
    df_raw = load_data('HEMS_Sample_Dataset.xlsx')
    df     = engineer_features(df_raw)
    try:
        load_models()
    except Exception:
        from logic.model import train_ac_predictor, train_peak_detector
        train_ac_predictor(df)
        train_peak_detector(df)
    df = predict_peaks(df, lookahead_hours=2)
    df = apply_control_logic(df)
    df = add_condensate_columns(df)
    df = add_cost_columns(df)
    return df

df        = load_pipeline()
kpis      = get_control_summary(df)
cost_kpis = get_cost_summary(df)
cond_kpis = get_condensate_summary(df)
scale     = get_scaling_summary(df)

st.sidebar.title(" HEMS Control Panel")
st.sidebar.markdown("**GSR Hackathon 2026 — Energy Track**")
st.sidebar.markdown("---")

dates    = sorted(df['timestamp'].dt.date.astype(str).unique())
sel_date = st.sidebar.selectbox("📅 Select Day", dates)

buildings = df['building_id'].unique().tolist()
sel_bldg  = st.sidebar.selectbox("🏠 Building", buildings)

st.sidebar.markdown("---")
st.sidebar.markdown("**Strategy**")
st.sidebar.markdown("🔵 **Pre-cool** — 2h before peak\n\n🔴 **Peak reduce** — 25% load cut\n\n🟠 **Comfort override** — dew point > 24°C")
st.sidebar.markdown("---")
st.sidebar.markdown("** Condensate Reuse**")
st.sidebar.markdown(f"Daily avg: **{cond_kpis['daily_avg_litres']} L/day**")
st.sidebar.markdown(f"Yearly est: **{cond_kpis['yearly_est_litres']:,.0f} L/yr**")

st.markdown("##  HEMS Smart A/C Optimization Dashboard")
st.markdown("Predictive demand response for residential buildings — Eastern Province, Saudi Arabia")
st.markdown("---")

def card(col, val, label, color=""):
    col.markdown(
        f'<div class="metric-card metric-card-{color}">'
        f'<div class="metric-value metric-value-{color}">{val}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="section-hdr"> Key Performance Indicators (30-Day)</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
card(c1, f"{kpis['peak_reduction_pct']}%",    "Peak Load Reduction")
card(c2, f"{kpis['energy_savings_pct']}%",    "Energy Savings")
card(c3, f"{kpis['comfort_pct']}%",           "Comfort Maintained")
card(c4, f"{kpis['total_saved_kwh']} kWh",    "Total kWh Saved")
card(c5, f"{kpis['peak_hours_reduced']} hrs", "Peak Hours Reduced")

st.markdown('<div class="section-hdr"> Cost Savings & Condensate Water Recovery</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
card(k1, f"{cost_kpis['monthly_saved_sar']} SAR",         "Monthly Bill Saved",      "amber")
card(k2, f"{cost_kpis['yearly_est_saved_sar']} SAR",      "Projected Annual Savings", "amber")
card(k3, f"{cost_kpis['daily_avg_saved_sar']} SAR",       "Avg Daily Saving",         "amber")
card(k4, f"{cond_kpis['daily_avg_litres']} L/day",        "Condensate Recovered",    "water")
card(k5, f"{cond_kpis['yearly_est_litres']:,.0f} L",      "Yearly Water Yield",      "water")

st.markdown("---")

day_df = df[
    (df['timestamp'].dt.date.astype(str) == sel_date) &
    (df['building_id'] == sel_bldg)
].sort_values('hour')

st.markdown('<div class="section-hdr"> Before vs After — Hourly A/C Load</div>', unsafe_allow_html=True)

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

col_l, col_r = st.columns(2)

with col_l:
    st.markdown('<div class="section-hdr"> Comfort Score (Hourly)</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-hdr"> Control Mode Distribution</div>', unsafe_allow_html=True)
    mode_cnt = day_df['control_mode'].value_counts().reset_index()
    mode_cnt.columns = ['Mode', 'Hours']
    color_map = {'normal':'#3a7d44','pre_cool':'#1a6cf5',
                 'peak_reduce':'#e04848','comfort_override':'#c97d1b'}
    fig_p = px.pie(mode_cnt, values='Hours', names='Mode',
                   color='Mode', color_discrete_map=color_map, hole=0.45)
    fig_p.update_layout(template='plotly_dark', height=290, margin=dict(l=20,r=20,t=10,b=20))
    st.plotly_chart(fig_p, use_container_width=True)

st.markdown('<div class="section-hdr"> Climate — Temperature & Dew Point</div>', unsafe_allow_html=True)
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

st.markdown('<div class="section-hdr"> Condensate Water Recovery (Hourly)</div>', unsafe_allow_html=True)
fig_w = go.Figure()
fig_w.add_trace(go.Bar(
    x=day_df['hour'], y=day_df['condensate_baseline_L'],
    name='Baseline condensate', marker_color='#64748b', opacity=0.6
))
fig_w.add_trace(go.Bar(
    x=day_df['hour'], y=day_df['condensate_optimized_L'],
    name='Recovered (optimized)', marker_color='#38bdf8', opacity=0.9
))
fig_w.update_layout(template='plotly_dark', barmode='group', height=250,
    xaxis_title='Hour', yaxis_title='Condensate (litres)',
    legend=dict(orientation='h', y=1.1),
    margin=dict(l=40,r=20,t=10,b=40))
st.plotly_chart(fig_w, use_container_width=True)

st.markdown('<div class="section-hdr"> Hourly Cost — Baseline vs Optimized (SAR)</div>', unsafe_allow_html=True)
fig_cost = go.Figure()
fig_cost.add_trace(go.Scatter(
    x=day_df['hour'], y=day_df['cost_baseline_sar'],
    name='Baseline cost (SAR)', line=dict(color='#e04848', dash='dash', width=2),
    mode='lines+markers'
))
fig_cost.add_trace(go.Scatter(
    x=day_df['hour'], y=day_df['cost_optimized_sar'],
    name='Optimized cost (SAR)', line=dict(color='#f5a623', width=2),
    mode='lines+markers', fill='tonexty', fillcolor='rgba(245,166,35,0.07)'
))
fig_cost.update_layout(template='plotly_dark', height=250,
    xaxis_title='Hour', yaxis_title='Cost (SAR)',
    legend=dict(orientation='h', y=1.1),
    margin=dict(l=40,r=20,t=10,b=40))
st.plotly_chart(fig_cost, use_container_width=True)

st.markdown('<div class="section-hdr"> Optimized Hourly Schedule</div>', unsafe_allow_html=True)

sched = build_daily_schedule(df, sel_date)
sched = sched[sched['hour'].isin(day_df['hour'])]

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
}).style.map(mode_color, subset=['Mode'])

st.dataframe(styled, use_container_width=True, height=420)

st.markdown('<div class="section-hdr"> 30-Day Daily Energy Savings</div>', unsafe_allow_html=True)

daily = df.groupby(df['timestamp'].dt.date.astype(str)).agg(
    Baseline=('ac_kwh', 'sum'),
    Optimized=('optimized_ac_kwh', 'sum'),
    Saved_SAR=('cost_saved_sar', 'sum'),
    Condensate_L=('condensate_optimized_L', 'sum')
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

# Scaling Impact Section 
st.markdown("---")
st.markdown('<div class="section-hdr"> Scaling Impact — Eastern Province, Saudi Arabia</div>', unsafe_allow_html=True)
st.markdown("*If this system were deployed across all residential buildings in the Eastern Province:*")

s1, s2, s3, s4 = st.columns(4)

def scale_box(col, val, title, sub):
    col.markdown(
        f'<div class="scale-box"><div class="scale-title">{title}</div>'
        f'<div class="scale-val">{val}</div>'
        f'<div class="scale-sub">{sub}</div></div>',
        unsafe_allow_html=True
    )

scale_box(s1, f"{scale['ep_gwh_saved_year']:,.1f} GWh",
          " Energy Saved / Year",
          "Across 850,000 Eastern Province homes")
scale_box(s2, f"{scale['ep_sar_saved_year_B']:.2f}B SAR",
          " Bill Savings / Year",
          "Total household savings, Eastern Province")
scale_box(s3, f"{scale['ep_mw_peak_reduced']:,.0f} MW",
          " Peak Demand Reduced",
          "Grid stress relief during peak hours")
scale_box(s4, f"{scale['ep_co2_saved_tonnes_year']:,.0f} t",
          " CO₂ Avoided / Year",
          f"At 0.64 kg CO₂/kWh (Saudi grid factor)")

st.markdown(f"""
<div style='background:#1a1a2e; border-radius:10px; padding:14px 20px; margin-top:10px; border:1px solid #2a2a4a;'>
  <span style='color:#aaa; font-size:0.85rem;'> Neighbourhood level (500 homes): </span>
  <span style='color:#c084fc; font-weight:600;'>{scale['hood_kwh_saved_day']:,.0f} kWh/day saved</span>
  <span style='color:#888;'> · </span>
  <span style='color:#c084fc; font-weight:600;'>{scale['hood_sar_saved_day']:,.0f} SAR/day</span>
  <span style='color:#888;'> · </span>
  <span style='color:#c084fc; font-weight:600;'>{scale['hood_mw_peak_reduced']*1000:.1f} kW peak demand cut</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(" **GSR Hackathon 2026 — Energy Track** &nbsp;|&nbsp; HEMS Data-Driven Demand Response &nbsp;|&nbsp; Noor · Hana · Hamza · Zainab")