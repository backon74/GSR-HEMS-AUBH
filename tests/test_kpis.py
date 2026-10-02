import json
import os
import re

import config
from evaluation import report


def _k(pipeline_outputs):
    return json.load(open('results/kpis.json'))


def test_schema_and_signs(pipeline_outputs):
    k = _k(pipeline_outputs)
    for key in ['data_source', 'headline', 'cost_sar', 'co2_kg', 'condensate_L_day', 'payback_months', 'scaling',
                'override_hours_by_reason', 'energy_change_open_loop_pct', 'sensitivity_range']:
        assert key in k
    assert 0 < k['headline']['peak_reduction_pct'] < 100
    assert k['cost_sar']['saved_per_year_flat'] < k['cost_sar']['saved_per_year_tou']
    assert 'baseline' in k['baseline_comfort_note'].lower()


def test_open_loop_saves_at_least_closed_loop(pipeline_outputs):
    k = _k(pipeline_outputs)
    assert k['energy_change_open_loop_pct'] <= k['energy_change_pct']
    assert k['kwh_recovery_per_day'] > 0


def test_scaling_uses_adoption_not_100pct(pipeline_outputs):
    sc = _k(pipeline_outputs)['scaling']['eastern_province_tou']
    assert set(sc) == {'10pct', '30pct', '50pct'}


def test_claims_have_sources(pipeline_outputs):
    txt = open('results/claims.md').read()
    rows = [l for l in txt.splitlines() if l.startswith('| ') and not l.startswith('| Claim') and '---' not in l]
    assert len(rows) > 20
    for l in rows:
        cells = [c.strip() for c in l.strip('|').split('|')]
        src = cells[3]
        assert re.fullmatch(r'(kpis|ablation)\.json:[\w.]+|E([1-9]|10)|assumption', src), l


def test_claims_values_match_kpis(pipeline_outputs):
    k = _k(pipeline_outputs)
    txt = open('results/claims.md').read()
    assert f"| {k['headline']['peak_reduction_pct']:g} |" in txt


def test_readme_numbers_match_kpis(pipeline_outputs):
    k = _k(pipeline_outputs)
    readme = open('README.md').read()
    assert report.readme_block(k) in readme
    assert f"**{k['headline']['peak_reduction_pct']}%**" in readme


def test_provenance_in_every_output(pipeline_outputs):
    import pandas as pd
    assert pd.read_csv('data/processed/optimized_schedule.csv')['data_source'].eq('synthetic_sample').all()
    assert pd.read_csv('results/sensitivity.csv')['data_source'].eq('synthetic_sample').all()
    for f in ['results/kpis.json', 'results/demo_day.json', 'results/ablation.json']:
        assert json.load(open(f))['data_source'] == 'synthetic_sample'
    assert 'synthetic_sample' in open('results/claims.md').read()
    assert 'synthetic_sample' in open('firmware/replay_table.h').read()


def test_evidence_register_complete(pipeline_outputs):
    txt = open('docs/EVIDENCE.md').read()
    for i in range(1, 11):
        assert f'| E{i} |' in txt


def test_ablation_states_ml_verdict(pipeline_outputs):
    a = json.load(open('results/ablation.json'))
    assert isinstance(a['rf_beats_naive'], bool) and a['verdict']


def test_no_stale_claims_in_readme():
    readme = open('README.md').read()
    for bad in ['30–40%', '~SAR 900M', 'maintained 100%', 'DHT22']:
        assert bad not in readme
