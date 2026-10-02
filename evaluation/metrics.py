"""KPI computation (7.7). Single output: results/kpis.json. Every number here is a simulation result on
the synthetic sample unless tagged otherwise in claims.md."""
import numpy as np
import pandas as pd

import config
from logic.condensate import estimate_condensate


def _tariff(hours, kind):
    if kind == 'flat':
        return np.full(len(hours), config.TARIFF_FLAT_SAR)
    return np.where(np.isin(hours, config.PEAK_HOURS), config.TARIFF_PEAK_SAR, config.TARIFF_OFFPEAK_SAR)


def cost_per_day(s, kind):
    t = _tariff(s['hour'].to_numpy(), kind)
    n = s['date'].nunique()
    base = float((s['kwh_baseline'] * t).sum() / n)
    opt = float((s['kwh_optimized'] * t).sum() / n)
    return base, opt


def core_metrics(s):
    """Compact metrics used by KPIs and by every sensitivity / ablation row."""
    n_days = s['date'].nunique()
    peak = s[s['hour'].isin(config.PEAK_HOURS)]
    pb, po = peak['kwh_baseline'].sum(), peak['kwh_optimized'].sum()
    tb, to_ = s['kwh_baseline'].sum(), s['kwh_optimized'].sum()
    rec = s['kwh_recovery'].sum()
    out = {
        'days': int(n_days),
        'peak_reduction_pct': round(float((pb - po) / pb * 100), 2),
        'peak_kw_reduction_avg': round(float((pb - po) / len(peak)), 4),
        'energy_change_pct': round(float((to_ - tb) / tb * 100), 2),
        'energy_change_open_loop_pct': round(float((to_ - rec - tb) / tb * 100), 2),
        'kwh_baseline_per_day': round(float(tb / n_days), 3),
        'kwh_optimized_per_day': round(float(to_ / n_days), 3),
        'kwh_recovery_per_day': round(float(rec / n_days), 3),
        'comfort_pct': round(float((s['indoor_temp_est_c'] <= config.COMFORT_T_MAX).mean() * 100), 2),
        'max_indoor_c': round(float(s['indoor_temp_est_c'].max()), 3),
        'hours_above_limit': int((s['indoor_temp_est_c'] > config.COMFORT_T_MAX).sum()),
        'override_hours': int((s['override_reason'] != 'none').sum()),
    }
    end = s[s['hour'] == max(config.PEAK_HOURS) + 1]
    out['mean_rise_at_peak_end_c'] = round(float(end['indoor_rise_c'].mean()), 3)
    for kind in ('tou', 'flat'):
        b, o = cost_per_day(s, kind)
        out[f'sar_saved_per_day_{kind}'] = round(b - o, 4)
        out[f'sar_saved_per_year_{kind}'] = round((b - o) * config.ANNUAL_COOLING_DAYS, 2)
    return out


def _payback(cost, yearly):
    return round(cost / yearly * 12, 1) if yearly > 0 else None


def condensate(s):
    n = s['date'].nunique()
    b = sum(estimate_condensate(a, h, t) for a, h, t in zip(s['kwh_baseline'], s['humidity'], s['temp']))
    o = sum(estimate_condensate(a, h, t) for a, h, t in zip(s['kwh_optimized'], s['humidity'], s['temp']))
    return round(b / n, 2), round(o / n, 2)


def compute_kpis(s, sens=None, provenance=None, forecast_info=None):
    m = core_metrics(s)
    n_days = m['days']
    saved_kwh_day = m['kwh_baseline_per_day'] - m['kwh_optimized_per_day']
    cb, co = condensate(s)
    yr_tou, yr_flat = m['sar_saved_per_year_tou'], m['sar_saved_per_year_flat']
    lo, hi = config.DEVICE_COST_RANGE_SAR

    def scale(yearly_home, kwh_home_year, kw_home):
        rows = {}
        for a in config.ADOPTION:
            homes = config.EP_RESIDENTIAL_HOMES * a
            rows[f'{int(a * 100)}pct'] = {
                'homes': int(homes),
                'sar_per_year_million': round(yearly_home * homes / 1e6, 1),
                'gwh_per_year': round(kwh_home_year * homes / 1e6, 1),
                'peak_mw_reduced': round(kw_home * homes / 1000, 0),
                'co2_tonnes_per_year': round(kwh_home_year * homes * config.CO2_KG_PER_KWH / 1000, 0),
            }
        return rows

    kwh_year = saved_kwh_day * config.ANNUAL_COOLING_DAYS
    scaling = {
        'one_home': {'kwh_saved_per_day': round(saved_kwh_day, 3), 'peak_kw_reduced': m['peak_kw_reduction_avg']},
        'neighbourhood_500_homes': {
            'kwh_saved_per_day': round(saved_kwh_day * config.AVG_HOMES_PER_HOOD, 1),
            'peak_mw_reduced': round(m['peak_kw_reduction_avg'] * config.AVG_HOMES_PER_HOOD / 1000, 3),
            'sar_per_year_tou': round(yr_tou * config.AVG_HOMES_PER_HOOD, 0),
            'sar_per_year_flat': round(yr_flat * config.AVG_HOMES_PER_HOOD, 0)},
        'eastern_province_tou': scale(yr_tou, kwh_year, m['peak_kw_reduction_avg']),
        'eastern_province_flat': scale(yr_flat, kwh_year, m['peak_kw_reduction_avg']),
        'ep_residential_homes': config.EP_RESIDENTIAL_HOMES,
        'ep_residential_homes_status': 'UNRESOLVED assumption: code 850,000 vs deck 1.5M; needs citation',
        'note': 'Adoption-scaled scenarios, not forecasts. Peak MW assumes all adopters peak together.',
    }
    modes = s['actual_mode'].value_counts().to_dict()
    kpis = {
        'data_source': (provenance or {}).get('data_source', config.DATA_SOURCE),
        'provenance': provenance or {},
        'scope': {'days': n_days, 'hours': int(len(s)), 'profile': config.DEFAULT_PROFILE,
                  'planner': config.PLANNER, 'plan_basis': 'rf (train days are in-sample, see forecast block)',
                  'type': 'simulation on synthetic sample; indoor temperature is modelled'},
        'headline': {
            'peak_reduction_pct': m['peak_reduction_pct'],
            'energy_change_pct_closed_loop': m['energy_change_pct'],
            'comfort_pct_indoor_modelled': m['comfort_pct'],
        },
        **m,
        'baseline_comfort_note': 'Baseline comfort is 100% by construction (baseline assumed to hold T_SET). '
                                 'Report SmartCool indoor excursions, not "100% comfort maintained".',
        'override_hours_by_reason': s['override_reason'].value_counts().to_dict(),
        'mode_counts': {k: int(v) for k, v in modes.items()},
        'cost_sar': {
            'per_day_tou': {'baseline': round(cost_per_day(s, 'tou')[0], 3), 'optimized': round(cost_per_day(s, 'tou')[1], 3)},
            'per_day_flat': {'baseline': round(cost_per_day(s, 'flat')[0], 3), 'optimized': round(cost_per_day(s, 'flat')[1], 3)},
            'saved_per_year_tou': yr_tou, 'saved_per_year_flat': yr_flat,
            'annualisation_days': config.ANNUAL_COOLING_DAYS,
            'flat_tariff_note': "Under today's flat 0.18 SAR/kWh the load shift itself saves nothing; only net kWh does.",
        },
        'co2_kg': {'saved_per_day': round(saved_kwh_day * config.CO2_KG_PER_KWH, 3),
                   'saved_per_year': round(kwh_year * config.CO2_KG_PER_KWH, 1),
                   'factor_kg_per_kwh': config.CO2_KG_PER_KWH},
        'condensate_L_day': {'baseline': cb, 'optimized': co,
                             'note': 'Formula unconfirmed (I9 bug fixed: temp not dew point passed). Order of magnitude only.'},
        'payback_months': {
            'device_cost_sar': config.DEVICE_COST_SAR, 'device_cost_range_sar': [lo, hi],
            'tou': _payback(config.DEVICE_COST_SAR, yr_tou), 'flat': _payback(config.DEVICE_COST_SAR, yr_flat),
            'tou_at_cost_range': [_payback(lo, yr_tou), _payback(hi, yr_tou)],
            'flat_at_cost_range': [_payback(lo, yr_flat), _payback(hi, yr_flat)],
            'note': 'Computed from our savings; device cost is an assumption.'},
        'scaling': scaling,
    }
    if forecast_info:
        kpis['forecast'] = forecast_info
    if sens is not None and len(sens):
        typ = sens[(sens['profile'] == config.DEFAULT_PROFILE) & (sens['rise_max_c'] == config.PLAN_RISE_MAX_C)
                   & (sens['k_rec'] == config.K_REC)].iloc[0]
        rng = {}
        for c in ['peak_cut_pct', 'energy_change_pct', 'hours_above_limit', 'override_hours',
                  'sar_per_year_tou', 'sar_per_year_flat']:
            rng[c] = {'typical': float(typ[c]), 'min': float(sens[c].min()), 'max': float(sens[c].max())}
        kpis['sensitivity_range'] = {
            'grid': 'profile {tight,typical,leaky} x rise limit {1.0,1.5,2.0} x K_REC {0.25,0.5,1.0}',
            'cells': int(len(sens)), 'metrics': rng}
    return kpis
