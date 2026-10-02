"""Decision engine (7.5): plan_day() + step(). run_batch() is literally a loop over step().

Python plans, the device executes and protects, the dashboard observes.
"""
import numpy as np
import pandas as pd

import config
from logic import indoor_model as im
from logic.planner import plan_day, planned_opt  # noqa: F401  (re-exported)

DECISION_COLUMNS = [
    'planned_mode', 'actual_mode', 'override_reason', 'indoor_temp_est_c', 'indoor_rise_c',
    'predicted_indoor_next_c', 'kwh_baseline', 'kwh_optimized', 'kwh_recovery', 'setpoint_adj',
    'fan_pwm', 'led', 'tariff_period', 'predicted_kwh', 'data_source', 'scenario',
    'outdoor_dew_flag', 'reason', 'trace',
]
_SETPOINT_ADJ = {'normal': 0.0, 'pre_cool': -1.0, 'peak_reduce': 1.0, 'comfort_override': -0.5}


def initial_state(delta0=0.0):
    return {'delta': float(delta0)}


def tariff_period(hour):
    return 'peak' if hour in config.PEAK_HOURS else 'offpeak'


def fan_for(mode, temp=None, dew=None):
    if mode == 'normal' and temp is not None and dew is not None \
            and dew < config.FAN_DRY_DEW_C and temp < config.FAN_DRY_TEMP_C:
        return config.FAN_PWM_NORMAL_DRY
    return config.FAN_PWM[mode]


def scenario_scale(scenario, hour):
    if scenario and hour in scenario.get('hours', range(24)):
        return float(scenario.get('load_scale', 1.0))
    return 1.0


def step(state, hour_row, plan_row, scenario=None, params=None):
    """One simulated hour. Returns (decision dict, new state)."""
    params = params or im.get_params()
    hour = int(hour_row['hour'])
    delta = float(state['delta'])
    t_in = params['t_set'] + delta
    scale = scenario_scale(scenario, hour)
    base = float(hour_row['ac_kwh']) * scale
    planned = plan_row['planned_mode']
    trace = [f"plan: {planned} (cut {plan_row['cut_frac']:.2f}, boost {plan_row['boost_kwh']:.2f} kWh)"]
    if scale != 1.0:
        trace.append(f"scenario {scenario['name']}: baseline x{scale:.2f}")

    recovering = False
    if planned == 'normal' and plan_row['post_window'] and delta > config.RECOVERY_EPS_C:
        opt = float(im.recovery_opt(delta, base, params))
        recovering = True
        trace.append(f"recovery: indoor +{delta:.2f} C above setpoint, thermostat pulls back")
    else:
        opt = float(planned_opt(planned, base, plan_row['cut_frac'], plan_row['boost_kwh']))
    pred_next_delta = float(im.predict_next(delta, base, opt, params))
    pred_next_c = params['t_set'] + pred_next_delta

    actual = planned
    reason = 'none'
    why = f"{planned}: following plan"
    reducing = opt < base - 1e-12
    if reducing and (t_in > config.COMFORT_T_MAX or pred_next_c > config.COMFORT_T_MAX):
        actual, reason = 'comfort_override', 'indoor_temp'
        opt = base
        recovering = False
        why = (f"override: predicted indoor {pred_next_c:.1f} C would exceed {config.COMFORT_T_MAX:.1f} C"
               if pred_next_c > config.COMFORT_T_MAX else f"override: indoor {t_in:.1f} C above limit")
        trace.append(f"gate: T_in {t_in:.2f}, next {pred_next_c:.2f} > {config.COMFORT_T_MAX:.1f} -> hold cooling")
    else:
        trace.append(f"gate: next {pred_next_c:.2f} C <= {config.COMFORT_T_MAX:.1f} C ok")
        if recovering:
            why = "recovery: restoring setpoint after the reduction window"

    kwh_rec = opt - base if recovering else 0.0
    new_delta = float(im.predict_next(delta, base, opt, params))
    temp, dew = hour_row.get('temp'), hour_row.get('dew_point')
    led = config.MODE_LED[actual]
    dec = {
        'planned_mode': planned, 'actual_mode': actual, 'override_reason': reason,
        'indoor_temp_est_c': round(t_in, 4), 'indoor_rise_c': round(delta, 4),
        'predicted_indoor_next_c': round(pred_next_c, 4),
        'kwh_baseline': round(base, 4), 'kwh_optimized': round(opt, 4), 'kwh_recovery': round(kwh_rec, 4),
        'setpoint_adj': _SETPOINT_ADJ[actual], 'fan_pwm': fan_for(actual, temp, dew), 'led': led,
        'tariff_period': tariff_period(hour), 'predicted_kwh': hour_row.get('predicted_kwh', np.nan),
        'data_source': hour_row.get('data_source', config.DATA_SOURCE),
        'scenario': scenario['name'] if scenario else 'none',
        'outdoor_dew_flag': bool(dew is not None and dew >= config.DEW_POINT_UNCOMFORTABLE),
        'reason': why, 'trace': ' | '.join(trace),
    }
    return dec, {'delta': new_delta}


def run_batch(df, plans, scenario=None, params=None, scenario_dates=None):
    """Loop over step(). df: hourly rows (date, hour, ac_kwh, ...). plans: {date_str: plan DataFrame}.
    scenario applies to scenario_dates (iterable of date strings) or all days when None."""
    params = params or im.get_params()
    out = []
    for date, day in df.sort_values('timestamp').groupby(df['timestamp'].dt.date.astype(str), sort=True):
        plan = plans[date].set_index('hour')
        state = initial_state()
        sc = scenario if (scenario_dates is None or date in set(scenario_dates)) else None
        for _, row in day.iterrows():
            dec, state = step(state, row, plan.loc[int(row['hour'])].to_dict(), sc, params)
            out.append({'timestamp': row['timestamp'], 'date': date, 'hour': int(row['hour']), **dec})
    res = pd.DataFrame(out)
    keep = [c for c in ['timestamp', 'building_id', 'temp', 'humidity', 'dew_point', 'solar', 'ac_kwh', 'is_peak',
                        'predicted_peak_proba'] if c in df.columns]
    base = df[keep].copy()
    res = base.merge(res, on='timestamp', how='inner') if len(res) else res
    return add_legacy_aliases(res)


def add_legacy_aliases(df):
    """Aliases for the old Streamlit dashboard until it is replaced."""
    df = df.copy()
    df['control_mode'] = df['actual_mode']
    df['optimized_ac_kwh'] = df['kwh_optimized']
    df['ac_saved_kwh'] = (df['kwh_baseline'] - df['kwh_optimized']).clip(lower=0).round(4)
    span = config.COMFORT_T_MAX - config.T_SET
    df['comfort_score'] = ((config.COMFORT_T_MAX - df['indoor_temp_est_c']) / span).clip(0, 1).round(3)
    df['safe_to_reduce'] = df['predicted_indoor_next_c'] <= config.COMFORT_T_MAX
    df['predicted_peak'] = (df['tariff_period'] == 'peak').astype(int)
    df['pre_cool_window'] = (df['planned_mode'] == 'pre_cool').astype(int)
    return df


# ── Dashboard payload (9) ────────────────────────────────────────────────────
_CTX = {}


def _context():
    if 'fdf' not in _CTX:
        from logic import runner
        from logic.load_data import load_data
        df = load_data(verbose=False)
        fdf, _, split = runner.prepare(df)
        _CTX.update(fdf=fdf, split=split, prov=df.attrs['provenance'])
    return _CTX


def get_payload(date, hour, scenario=None, profile=None):
    """Everything the dashboard shows for (date, hour). Each field is tagged measured / replayed / modelled.
    'measured' fields come from the device and are None here; the dashboard fills them from telemetry."""
    from logic import runner
    ctx = _context()
    date = str(date)
    sched, plans, params = runner.build_schedule(ctx['fdf'], scenario=scenario, profile=profile, days=[date])
    day = sched[sched['date'] == date].sort_values('hour')
    row = day[day['hour'] == hour].iloc[0]
    upto = day[day['hour'] <= hour]
    tou_t = lambda h: config.TARIFF_PEAK_SAR if h in config.PEAK_HOURS else config.TARIFF_OFFPEAK_SAR
    from logic.condensate import estimate_condensate
    pk = upto[upto['hour'].isin(config.PEAK_HOURS)]
    saved = float((upto['kwh_baseline'] - upto['kwh_optimized']).sum())
    sar = float(sum((r.kwh_baseline - r.kwh_optimized) * tou_t(r.hour) for r in upto.itertuples()))
    cond_opt = float(sum(estimate_condensate(r.kwh_optimized, r.humidity, r.temp) for r in upto.itertuples()))
    tag = lambda v, t, e=None: {'value': v, 'tag': t, **({'evidence': e} if e else {})}
    return {
        'date': date, 'hour': int(hour), 'data_source': row['data_source'],
        'source_badge': 'SIM' if scenario else 'REPLAY',
        'scenario': tag(scenario, 'modelled') if scenario else None,
        'scenario_badge': 'SCENARIO' if scenario else None,
        'device': {k: tag(None, 'measured') for k in ('temp', 'rh', 'dew', 'fan_duty', 'sensor_ok')},
        'replayed': {'outdoor_temp': tag(float(row['temp']), 'replayed'), 'outdoor_rh': tag(float(row['humidity']), 'replayed'),
                     'outdoor_dew': tag(float(row['dew_point']), 'replayed'), 'solar': tag(float(row['solar']), 'replayed'),
                     'kwh_baseline': tag(float(row['kwh_baseline']), 'replayed')},
        'modelled': {
            'planned_mode': tag(row['planned_mode'], 'modelled'), 'actual_mode': tag(row['actual_mode'], 'modelled'),
            'override_reason': tag(row['override_reason'], 'modelled'), 'reason': tag(row['reason'], 'modelled'),
            'indoor_temp_est_c': tag(float(row['indoor_temp_est_c']), 'modelled'),
            'predicted_indoor_next_c': tag(float(row['predicted_indoor_next_c']), 'modelled'),
            'kwh_optimized': tag(float(row['kwh_optimized']), 'modelled'),
            'kwh_recovery': tag(float(row['kwh_recovery']), 'modelled'),
            'predicted_kwh': tag(float(row['predicted_kwh']), 'modelled'),
            'predicted_peak_proba': tag(float(row['predicted_peak_proba']), 'modelled') if 'predicted_peak_proba' in row and row['predicted_peak_proba'] == row['predicted_peak_proba'] else None,
            'fan_pwm': tag(int(row['fan_pwm']), 'modelled'), 'led': tag(row['led'], 'modelled'),
            'tariff_period': tag(row['tariff_period'], 'modelled'),
        },
        'decision_trace': row['trace'].split(' | '),
        'ribbon': [{'hour': int(r.hour), 'planned_mode': r.planned_mode, 'actual_mode': r.actual_mode,
                    'now': int(r.hour) == int(hour)} for r in day.itertuples()],
        'counters': {'kwh_saved_net': tag(round(saved, 3), 'modelled'), 'sar_saved_tou': tag(round(sar, 3), 'modelled'),
                     'peak_kwh_cut': tag(round(float((pk['kwh_baseline'] - pk['kwh_optimized']).sum()), 3), 'modelled'),
                     'condensate_L': tag(round(cond_opt, 3), 'modelled'),
                     'overrides': tag(int((upto['override_reason'] != 'none').sum()), 'modelled')},
        'evidence': {'cop': {'id': 'E5', 'type': 'lab', 'value': config.COP},
                     'humidity_effect': {'id': 'E3', 'type': 'field (1990, abstract-level)',
                                         'note': 'direction of effect; not demonstrated by the synthetic sample'}},
    }
