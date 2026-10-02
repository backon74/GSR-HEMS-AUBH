"""First-order indoor temperature layer (7.3).

Linearised around the observed baseline, which is assumed to hold T_SET.
Only the change in A/C energy moves indoor temperature:
    a = exp(-dt/tau)
    D[t+1] = a*D[t] + g*tau*(1-a)*(base[t] - opt[t])
    T_in[t] = T_SET + D[t]
Exact discretisation for a load held constant over each hour.

Optional DHT calibration: if config.PROFILE_OVERRIDE_PATH exists (from tools/fit_tau.py
--write-override), tau/g from that file replace the assumed HOUSE_PROFILES entry. That
improves this physics layer only — it does not change ML forecast FEATURES.
"""
import json
import os

import numpy as np
import pandas as pd

import config


def load_profile_override(path=None):
    """Return override dict or None. Keys: tau, g (optional), c_th (optional), data_source, label."""
    path = path or config.PROFILE_OVERRIDE_PATH
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_params(profile=None, **overrides):
    name = profile or config.DEFAULT_PROFILE
    p = dict(config.HOUSE_PROFILES[name])
    p.update({"t_set": config.T_SET, "k_rec": config.K_REC, "dt": config.DT_H,
              "profile": name, "profile_label": "assumed"})
    ov = load_profile_override()
    if ov and ov.get('tau') is not None:
        p['tau'] = float(ov['tau'])
        if ov.get('g') is not None:
            p['g'] = float(ov['g'])
        if ov.get('c_th') is not None:
            p['c_th'] = float(ov['c_th'])
        p['profile_label'] = ov.get('label', 'own-measured')
        p['data_source'] = ov.get('data_source', 'own_logger')
    p.update(overrides)
    return p


def _coef(params):
    a = np.exp(-params.get("dt", config.DT_H) / params["tau"])
    return a, params["g"] * params["tau"] * (1.0 - a)


def predict_next(delta, base, opt, params):
    """Delta after one hour. Works on scalars or numpy arrays."""
    a, b = _coef(params)
    return a * np.asarray(delta, dtype=float) + b * (np.asarray(base, dtype=float) - np.asarray(opt, dtype=float))


def recovery_opt(delta, base, params, ac_max=None):
    """Thermostat pull-back after the reduction window (D7). Array-safe."""
    ac_max = config.AC_MAX_KWH if ac_max is None else ac_max
    delta = np.asarray(delta, dtype=float)
    base = np.asarray(base, dtype=float)
    want = np.minimum(ac_max, base + params.get("k_rec", config.K_REC) * delta / params["g"])
    return np.where(delta > config.RECOVERY_EPS_C, np.maximum(want, base), base)


def simulate_indoor(base_kwh, opt_kwh, params, delta0=0.0):
    base = np.asarray(base_kwh, dtype=float)
    opt = np.asarray(opt_kwh, dtype=float)
    delta = np.empty(len(base))
    d = float(delta0)
    for t in range(len(base)):
        delta[t] = d
        d = float(predict_next(d, base[t], opt[t], params))
    return pd.DataFrame({"T_in": params.get("t_set", config.T_SET) + delta, "delta": delta})
