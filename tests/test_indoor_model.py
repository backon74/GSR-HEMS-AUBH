import numpy as np

from logic import indoor_model as im


def P(profile='typical'):
    return im.get_params(profile)


def test_zero_control_stays_at_setpoint():
    base = np.random.default_rng(0).uniform(0, 3.6, 24)
    r = im.simulate_indoor(base, base, P())
    assert np.allclose(r['delta'], 0.0)
    assert np.allclose(r['T_in'], 24.0)


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
