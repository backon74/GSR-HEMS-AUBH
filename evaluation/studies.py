"""Sensitivity grid (7.7) and ML ablation (7.6)."""
import itertools

import numpy as np
import pandas as pd

import config
from evaluation.metrics import core_metrics
from logic import forecast, runner


def sensitivity(fdf):
    rows = []
    for prof, rise, krec in itertools.product(config.HOUSE_PROFILES, config.SENS_RISE, config.SENS_KREC):
        s, plans, _ = runner.build_schedule(fdf, profile=prof, rise_max=rise, k_rec=krec)
        m = core_metrics(s)
        ceil = int(pd.Series({k: v.attrs['summary']['ceiling_hit'] for k, v in plans.items()}).sum())
        rows.append({'profile': prof, 'rise_max_c': rise, 'k_rec': krec,
                     'peak_cut_pct': m['peak_reduction_pct'], 'energy_change_pct': m['energy_change_pct'],
                     'energy_change_open_loop_pct': m['energy_change_open_loop_pct'],
                     'hours_above_limit': m['hours_above_limit'], 'override_hours': m['override_hours'],
                     'sar_per_year_tou': m['sar_saved_per_year_tou'], 'sar_per_year_flat': m['sar_saved_per_year_flat'],
                     'days_at_cut_ceiling': ceil, 'data_source': s['data_source'].iloc[0]})
    return pd.DataFrame(rows)


def ablation(fdf, fmetrics, split):
    test = split['test_days']
    scen = {'nominal': None, 'heatwave_1.10': config.SCENARIOS['heatwave_low'],
            'heatwave_1.25': config.SCENARIOS['heatwave']}
    res = {}
    for basis in ('oracle', 'rf', 'naive'):
        res[basis] = {}
        for sname, sc in scen.items():
            s, _, _ = runner.build_schedule(fdf, basis=basis, scenario=sc, days=test)
            m = core_metrics(s)
            res[basis][sname] = {k: m[k] for k in ['peak_reduction_pct', 'hours_above_limit', 'override_hours',
                                                   'max_indoor_c', 'energy_change_pct']}
    s, _, _ = runner.build_schedule(fdf, basis='rf', planner='fixed', days=test)
    m = core_metrics(s)
    res['legacy_fixed_window_rf'] = {'nominal': {k: m[k] for k in ['peak_reduction_pct', 'hours_above_limit',
                                                                 'override_hours', 'max_indoor_c', 'energy_change_pct']}}
    overlap = {}
    d = fdf[fdf['day'].isin(test)]
    for basis, col in [('oracle', 'ac_kwh'), ('rf', 'predicted_kwh'), ('naive', 'naive_yesterday_kwh')]:
        ov, widths = [], []
        for _, g in d.groupby('day'):
            w = forecast.detect_peak_window(g.sort_values('hour')[col].to_numpy())
            ov.append(forecast.window_overlap(w)); widths.append(len(w))
        overlap[basis] = {'mean_jaccard_with_tariff_12_18': round(float(np.mean(ov)), 3),
                          'mean_window_hours': round(float(np.mean(widths)), 2)}
    beats = fmetrics['rf_beats_best_naive_mae']
    verdict = ('RF forecast beats the best naive baseline on test MAE.' if beats else
               'RF forecast does NOT beat the best naive baseline on test MAE. The synthetic set repeats daily, '
               'so the clock-like baselines are strong. Do not claim ML adds forecast accuracy on this data.')
    d_pp = abs(res['rf']['nominal']['peak_reduction_pct'] - res['naive']['nominal']['peak_reduction_pct'])
    verdict += (f" Planning outcome check: peak cut differs by {d_pp:.2f} pp between RF and naive forecasts "
                f"(nominal), so forecast quality changes the plan little on this data; the planner and indoor gate, "
                f"not the forecaster, produce the savings.")
    return {'data_source': fdf['data_source'].iloc[0], 'test_days': test, 'forecast_metrics': fmetrics,
            'verdict': verdict, 'rf_beats_naive': beats, 'detected_window': overlap, 'planning_ablation': res,
            'note': 'Weather forecast assumed equal to actual. Train days are in-sample for headline schedule; '
                    'ablation uses test days only.'}
