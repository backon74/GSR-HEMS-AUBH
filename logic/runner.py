"""Glue: forecast -> plan per day -> engine.run_batch. One place that builds a schedule."""
import pandas as pd

import config
from logic import engine, forecast, indoor_model as im
from logic.planner import plan_day

_BASIS_COL = {'rf': 'predicted_kwh', 'oracle': 'ac_kwh', 'naive': 'naive_yesterday_kwh'}


def prepare(df):
    """Forecast frame (days with 24 h history) + forecaster + split info."""
    return forecast.forecast_frame(df)


def build_schedule(fdf, basis='rf', profile=None, rise_max=None, k_rec=None, planner=None,
                   scenario=None, scenario_dates=None, days=None):
    over = {}
    if k_rec is not None:
        over['k_rec'] = k_rec
    params = im.get_params(profile, **over)
    d = fdf if days is None else fdf[fdf['day'].isin(days)]
    col = _BASIS_COL[basis]
    plans = {}
    for day, g in d.groupby('day', sort=True):
        g = g.sort_values('hour')
        plans[day] = plan_day(day, g[col].to_numpy(), params, rise_max, planner)
    sched = engine.run_batch(d, plans, scenario, params, scenario_dates)
    summ = pd.DataFrame({k: v.attrs['summary'] for k, v in plans.items()}).T
    summ.index.name = 'date'
    sched = sched.merge(summ[['cut_frac', 'boost_kwh', 'n_pre', 'ceiling_hit']].rename(
        columns={'cut_frac': 'plan_cut_frac', 'boost_kwh': 'plan_boost_kwh', 'n_pre': 'plan_n_pre'}),
        left_on='date', right_index=True, how='left')
    sched['forecast_split'] = sched['timestamp'].map(d.set_index('timestamp')['forecast_split'])
    sched['plan_basis'] = basis
    return sched, plans, params
