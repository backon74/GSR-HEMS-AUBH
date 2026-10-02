import config
from logic import planner, runner
from logic import indoor_model as im


def test_constraint_satisfied_every_day(fdf):
    for rise in (1.0, 1.5, 2.0):
        s, _, _ = runner.build_schedule(fdf, basis='oracle', rise_max=rise)
        assert s['indoor_temp_est_c'].max() <= config.T_SET + rise + 1e-6
        # recovery tail after hour 18 is included in the check
        assert (s['indoor_rise_c'] <= rise + 1e-6).all()


def test_objective_monotone_in_rise_limit(fdf):
    cuts = [runner.build_schedule(fdf, basis='oracle', profile=p, rise_max=r)[0]
            .pipe(lambda s: (1 - s[s['hour'].isin(config.PEAK_HOURS)]['kwh_optimized'].sum() /
                             s[s['hour'].isin(config.PEAK_HOURS)]['kwh_baseline'].sum()))
            for p in ('typical',) for r in (1.0, 1.5, 2.0)]
    assert cuts[0] <= cuts[1] <= cuts[2]


def test_plan_shape_and_variable_precool(fdf):
    g = fdf[fdf['day'] == sorted(fdf['day'].unique())[0]].sort_values('hour')
    plan = planner.plan_day('d', g['ac_kwh'].to_numpy(), im.get_params())
    assert len(plan) == 24 and set(plan['planned_mode']) <= {'normal', 'pre_cool', 'peak_reduce'}
    assert (plan[plan['hour'].isin(config.PEAK_HOURS)]['planned_mode'] == 'peak_reduce').all()
    s, plans, _ = runner.build_schedule(fdf, basis='oracle')
    assert s['plan_n_pre'].nunique() > 1          # pre-cool length varies by day, not fixed 10-11


def test_fixed_planner_is_legacy_25pct(fdf):
    s, _, _ = runner.build_schedule(fdf, planner='fixed')
    assert s['plan_cut_frac'].max() <= config.PEAK_REDUCE_FRACTION + 1e-9
