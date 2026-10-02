"""Comfort-constrained pre-cool sizing (7.4). Vectorised grid search over (cut, boost, n_pre) per day.

The closed-loop simulation here uses the same indoor_model functions as engine.step, including the
thermostat-recovery term, so planned and executed numbers agree when forecast == actual.
"""
import itertools

import numpy as np
import pandas as pd

import config
from logic import indoor_model as im

PEAK_START = min(config.PEAK_HOURS)
PEAK_END = max(config.PEAK_HOURS)
_TARIFF = np.array([config.TARIFF_PEAK_SAR if h in config.PEAK_HOURS else config.TARIFF_OFFPEAK_SAR
                    for h in range(24)])


def mode_for_hour(hour, n_pre):
    if hour in config.PEAK_HOURS:
        return 'peak_reduce'
    if PEAK_START - n_pre <= hour < PEAK_START:
        return 'pre_cool'
    return 'normal'


def planned_opt(mode, base, cut, boost):
    """Planned A/C kWh for a mode (before gate/recovery). Array-safe."""
    base = np.asarray(base, dtype=float)
    peak = np.minimum(np.maximum(base * (1.0 - cut), config.MIN_AC_KWH), base)
    pre = np.minimum(np.maximum(config.AC_MAX_KWH, base), base + boost)
    return np.where(mode == 'peak_reduce', peak, np.where(mode == 'pre_cool', pre, base))


def _grid(cuts, boosts, npres):
    combos = list(itertools.product(cuts, boosts, npres))
    return (np.array([c[0] for c in combos]), np.array([c[1] for c in combos]), np.array([c[2] for c in combos]))


def _evaluate(base24, cut, boost, npre, params):
    n = len(cut)
    delta = np.zeros(n)
    max_d = np.zeros(n)
    net = np.zeros(n)
    cost = np.zeros(n)
    peak_cut = np.zeros(n)
    for h in range(24):
        b = base24[h]
        flat = np.full(n, b)
        if h in config.PEAK_HOURS:
            opt = planned_opt('peak_reduce', b, cut, 0.0)
            peak_cut += b - opt
        elif h > PEAK_END:
            opt = im.recovery_opt(delta, b, params)
        elif h < PEAK_START:
            pre = planned_opt('pre_cool', b, 0.0, boost)
            opt = np.where(PEAK_START - npre <= h, pre, flat)
        else:
            opt = flat
        net += opt - b
        cost += opt * _TARIFF[h]
        delta = im.predict_next(delta, b, opt, params)
        max_d = np.maximum(max_d, delta)
    peak_base = sum(base24[h] for h in config.PEAK_HOURS)
    red = peak_cut / peak_base if peak_base > 0 else np.zeros(n)
    return max_d, red, net, cost


def plan_day(date, forecast_kwh, params=None, rise_max=None, planner=None):
    params = params or im.get_params()
    rise_max = config.PLAN_RISE_MAX_C if rise_max is None else rise_max
    planner = planner or config.PLANNER
    base24 = np.asarray(forecast_kwh, dtype=float)
    assert len(base24) == 24, "plan_day needs 24 hourly values"

    if planner == 'fixed':
        cut, boost, npre = _grid([c for c in config.PLAN_CUT_GRID if c <= config.PEAK_REDUCE_FRACTION],
                                 (config.PRE_COOL_BOOST_KWH,), (config.FIXED_PRE_COOL_HOURS,))
    else:
        cut, boost, npre = _grid(config.PLAN_CUT_GRID, config.PLAN_BOOST_GRID, config.PLAN_NPRE_GRID)
    max_d, red, net, cost = _evaluate(base24, cut, boost, npre, params)
    ok = max_d <= rise_max + 1e-9
    idx = np.flatnonzero(ok)
    order = np.lexsort((np.round(cost[idx], 9), np.round(net[idx], 9), -np.round(red[idx], 6)))
    k = idx[order[0]]
    c, b, n = float(cut[k]), float(boost[k]), int(npre[k])
    rows = []
    for h in range(24):
        m = mode_for_hour(h, n) if (c > 0 or b > 0) else 'normal'
        if m == 'peak_reduce' and c == 0:
            m = 'normal'
        if m == 'pre_cool' and b == 0:
            m = 'normal'
        rows.append({'date': str(date), 'hour': h, 'planned_mode': m,
                     'cut_frac': c if m == 'peak_reduce' else 0.0,
                     'boost_kwh': b if m == 'pre_cool' else 0.0,
                     'post_window': h > PEAK_END, 'n_pre': n, 'forecast_kwh': float(base24[h])})
    plan = pd.DataFrame(rows)
    plan.attrs['summary'] = {
        'cut_frac': c, 'boost_kwh': b, 'n_pre': n, 'plan_max_rise_c': round(float(max_d[k]), 3),
        'plan_peak_cut_pct': round(float(red[k]) * 100, 2), 'plan_net_kwh': round(float(net[k]), 3),
        'ceiling_hit': bool(c >= config.PLAN_CUT_CEILING - 1e-9 and planner == 'grid'),
    }
    return plan
