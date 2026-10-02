"""python pipeline.py: one codebase produces every number and command (results/, firmware/, data/processed/)."""
import json
import os

import numpy as np
import pandas as pd

import config
from evaluation import exports, metrics, report, studies
from evaluation.evidence import evidence_md
from logic import forecast, runner
from logic.features import engineer_features, get_feature_columns
from logic.load_data import load_data


def _peak_proba(df):
    """Legacy peak classifier probability per hour (Analytics tab). Loads the artefact; never overwrites it."""
    feats = [f for f in get_feature_columns() if f != 'hour']
    dfe = engineer_features(df)
    try:
        import joblib
        model = joblib.load('logic/peak_detector.pkl')
    except Exception as e:  # missing or incompatible artefact: train in memory only
        print(f"[peak model] artefact unusable ({type(e).__name__}); training in memory, not saved")
        from logic.model import train_peak_detector
        model = train_peak_detector(dfe, save=False)
    return dict(zip(dfe['timestamp'], model.predict_proba(dfe[feats])[:, 1]))


def _json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def run_pipeline():
    for d in ('results', 'docs', 'firmware', 'data/processed'):
        os.makedirs(d, exist_ok=True)
    df = load_data()
    prov = df.attrs['provenance']
    fdf, _, split = runner.prepare(df)
    fdf['predicted_peak_proba'] = fdf['timestamp'].map(_peak_proba(df)).round(4)
    fm = forecast.forecast_metrics(fdf)

    sched, plans, _ = runner.build_schedule(fdf)
    sens = studies.sensitivity(fdf)
    abl = studies.ablation(fdf, fm, split)
    fi = {'metrics_test_days': fm, 'train_days': len(split['train_days']), 'test_days': len(split['test_days']),
          'note': 'Headline schedule uses RF day-ahead forecasts; train days are in-sample. Ablation uses test days only.'}
    kpis = metrics.compute_kpis(sched, sens, prov, fi)
    kpis['days_at_cut_ceiling'] = int(sched.groupby('date')['ceiling_hit'].first().sum())

    out = sched.copy()
    out['timestamp'] = out['timestamp'].astype(str)
    out.to_csv('data/processed/optimized_schedule.csv', index=False)
    sens.to_csv('results/sensitivity.csv', index=False)
    _json('results/kpis.json', kpis)
    _json('results/ablation.json', abl)
    demo = exports.select_demo_day(fdf)
    exports.write_all(demo)
    open('results/claims.md', 'w').write(report.claims_md(kpis, abl))
    open('docs/EVIDENCE.md', 'w').write(evidence_md())
    report.update_readme(kpis)

    h = kpis['headline']
    print(f"\n=== SmartCool KPIs ({prov['data_source']}, {kpis['scope']['days']} days) ===")
    print(f"  peak reduction        : {h['peak_reduction_pct']}%")
    print(f"  energy (closed-loop)  : {h['energy_change_pct_closed_loop']}%   open-loop {kpis['energy_change_open_loop_pct']}%")
    print(f"  comfort (modelled)    : {h['comfort_pct_indoor_modelled']}%   max indoor {kpis['max_indoor_c']} C")
    print(f"  modes                 : {kpis['mode_counts']}")
    print(f"  forecast              : {abl['verdict']}")
    print(f"  demo day              : {demo['date']} scenario={demo['scenario']['name'] if demo['scenario'] else 'none'} "
          f"loop {demo['loop_seconds']} s")
    return sched, kpis


if __name__ == '__main__':
    run_pipeline()
