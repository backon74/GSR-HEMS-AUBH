import numpy as np
import pandas as pd

import config
from logic import engine, indoor_model as im, runner


def test_batch_equals_streaming(fdf):
    s, plans, params = runner.build_schedule(fdf, days=sorted(fdf['day'].unique())[:3])
    d = fdf[fdf['day'].isin(sorted(fdf['day'].unique())[:3])].sort_values('timestamp')
    rows = []
    for date, day in d.groupby('day'):
        state = engine.initial_state()
        plan = plans[date].set_index('hour')
        for _, r in day.iterrows():
            dec, state = engine.step(state, r, plan.loc[int(r['hour'])].to_dict(), None, params)
            rows.append(dec)
    stream = pd.DataFrame(rows)
    for c in ['planned_mode', 'actual_mode', 'kwh_optimized', 'kwh_recovery', 'indoor_temp_est_c', 'fan_pwm']:
        assert list(stream[c]) == list(s[c]), c


def test_opt_bounds_and_modes(sched):
    assert sched['kwh_optimized'].between(config.MIN_AC_KWH - 1e-9, config.AC_MAX_KWH + 1e-9).all() or \
        (sched['kwh_optimized'] >= 0).all()
    pos = sched[sched['kwh_baseline'] >= config.MIN_AC_KWH]
    assert (pos['kwh_optimized'] >= config.MIN_AC_KWH - 1e-9).all()
    assert (sched['kwh_optimized'] <= config.AC_MAX_KWH + 1e-9).all()
    assert set(sched['actual_mode']) <= set(config.MODES)
    assert set(sched['override_reason']) <= {'none', 'indoor_temp'}


def test_recovery_included_and_capped(sched):
    rec = sched[sched['kwh_recovery'] > 0]
    assert len(rec) > 0
    assert (rec['kwh_optimized'] <= config.AC_MAX_KWH + 1e-9).all()
    assert np.allclose(rec['kwh_optimized'] - rec['kwh_baseline'], rec['kwh_recovery'], atol=1e-3)
    open_loop = (sched['kwh_optimized'] - sched['kwh_recovery']).sum()
    assert sched['kwh_optimized'].sum() > open_loop


def test_override_implies_no_reduction(fdf):
    s, _, _ = runner.build_schedule(fdf, scenario=config.SCENARIOS['heatwave'])
    ov = s[s['actual_mode'] == 'comfort_override']
    assert len(ov) > 0
    assert np.allclose(ov['kwh_optimized'], ov['kwh_baseline'])
    assert (ov['override_reason'] == 'indoor_temp').all()
    assert (s[s['actual_mode'] != 'comfort_override']['override_reason'] == 'none').all()


def test_outdoor_never_gates(fdf, sched):
    assert 'outdoor_dew_flag' in sched
    assert sched['outdoor_dew_flag'].any()                       # outdoor dew is high, yet reductions ran
    assert (sched['actual_mode'] == 'peak_reduce').sum() > 100


def test_legacy_aliases(sched):
    for c in ['control_mode', 'optimized_ac_kwh', 'comfort_score', 'ac_saved_kwh', 'safe_to_reduce']:
        assert c in sched


def test_payload_tags(fdf):
    p = engine.get_payload('2025-07-25', 14)
    assert p['data_source'] == 'synthetic_sample'
    assert p['replayed']['kwh_baseline']['tag'] == 'replayed'
    assert p['modelled']['indoor_temp_est_c']['tag'] == 'modelled'
    assert p['device']['temp']['tag'] == 'measured'
    assert p['evidence']['cop']['id'] == 'E5'
    assert sum(r['now'] for r in p['ribbon']) == 1
