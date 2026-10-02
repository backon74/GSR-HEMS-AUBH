import numpy as np

import config
from logic import indoor_model as im


def P(profile='typical'):
    return im.get_params(profile)


def test_get_params_assumed_without_override(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'PROFILE_OVERRIDE_PATH', str(tmp_path / 'missing.json'))
    p = im.get_params('typical')
    assert p['profile_label'] == 'assumed' and p['tau'] == config.HOUSE_PROFILES['typical']['tau']


def test_get_params_reads_dht_override(monkeypatch, tmp_path):
    path = tmp_path / 'house_profile_override.json'
    path.write_text('{"tau": 7.5, "g": 0.45, "c_th": 5.0, "data_source": "own_logger", "label": "own-measured"}')
    monkeypatch.setattr(config, 'PROFILE_OVERRIDE_PATH', str(path))
    p = im.get_params('typical')
    assert p['tau'] == 7.5 and p['profile_label'] == 'own-measured' and p['data_source'] == 'own_logger'


def test_constant_reduction_matches_closed_form():
    for prof in ('tight', 'typical', 'leaky'):
        p = P(prof)
        u, n = 0.8, 6
        base = np.full(24, 2.0)
        opt = base.copy()
        opt[:n] -= u
        r = im.simulate_indoor(base, opt, p)
        a = np.exp(-1.0 / p['tau'])
        assert abs(r['delta'][n] - p['g'] * p['tau'] * u * (1 - a ** n)) < 1e-9


def test_cut_raises_boost_lowers_and_is_stable():
    p = P()
    base = np.full(24, 2.0)
    cut = base.copy(); cut[5:10] -= 1
    boost = base.copy(); boost[5:10] += 1
    up = im.simulate_indoor(base, cut, p)['delta'].to_numpy()
    dn = im.simulate_indoor(base, boost, p)['delta'].to_numpy()
    assert (up[6:11] > 0).all() and (dn[6:11] < 0).all()
    assert (np.diff(up[5:10]) > 0).all()          # monotone rise under constant cut
    assert (np.diff(up[10:]) < 0).all()           # monotone decay afterwards
    assert up.max() < p['g'] * p['tau'] * 1.0     # bounded by steady state
    assert abs(up[-1]) < up.max()


def test_deterministic():
    base = np.linspace(0.5, 3, 24)
    a = im.simulate_indoor(base, base * 0.7, P())
    b = im.simulate_indoor(base, base * 0.7, P())
    assert a.equals(b)


def test_recovery_bounds():
    p = P()
    d = np.array([0.0, 0.04, 0.5, 2.0])
    out = im.recovery_opt(d, np.full(4, 3.0), p)
    assert (out <= 3.6 + 1e-12).all() and out[0] == 3.0 and out[1] == 3.0 and out[2] > 3.0
