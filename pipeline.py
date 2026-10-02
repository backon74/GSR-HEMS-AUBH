"""python pipeline.py: one codebase produces every number and command (results/, firmware/, data/processed/).

Layers (do not conflate):
  outdoor weather -> day-ahead RF (in-memory, seeded) -> planner + indoor physics
  -> results/ + demo_day.json + firmware/replay_table.h + HARDWARE_CONTRACT.md
  -> bridge/serial_bridge.py -> ESP32 (DHT = live override only; optional tau override file)
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

import config
from evaluation import exports, metrics, report, studies
from evaluation.evidence import evidence_md
from logic import forecast, indoor_model as im, runner
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


def _gate_forecast(fm):
    """Warn loudly (and fail if SMARTCOOL_STRICT_FORECAST=1) when RF loses to best naive MAE."""
    if fm.get('rf_beats_best_naive_mae'):
        return
    msg = (f"Day-ahead RF MAE {fm['rf']['mae_kwh']} kWh does not beat best naive "
           f"(yesterday={fm['naive_same_hour_yesterday']['mae_kwh']}, "
           f"hod={fm['naive_hour_of_day_mean']['mae_kwh']}) on test days.")
    if os.environ.get('SMARTCOOL_STRICT_FORECAST') == '1':
        raise RuntimeError(msg)
    warnings.warn(msg, stacklevel=2)
    print(f"[forecast gate] WARNING: {msg}")


def run_pipeline():
    for d in ('results', 'docs', 'firmware', 'data/processed'):
        os.makedirs(d, exist_ok=True)
    df = load_data()
    prov = dict(df.attrs['provenance'])
    params = im.get_params()
    if params.get('profile_label') == 'own-measured':
        prov['indoor_profile'] = 'own-measured'
        prov['indoor_tau_h'] = params['tau']
        print(f"[indoor] using DHT-calibrated tau={params['tau']} h (physics only; RF still outdoor)")
    else:
        prov['indoor_profile'] = 'assumed'
    fdf, _, split = runner.prepare(df)
    fdf['predicted_peak_proba'] = fdf['timestamp'].map(_peak_proba(df)).round(4)
    fm = forecast.forecast_metrics(fdf)
    _gate_forecast(fm)

    sched, plans, _ = runner.build_schedule(fdf)
    sens = studies.sensitivity(fdf)
    abl = studies.ablation(fdf, fm, split)
    fi = {'metrics_test_days': fm, 'train_days': len(split['train_days']), 'test_days': len(split['test_days']),
          'note': 'Headline schedule uses RF day-ahead forecasts (outdoor features); train days are in-sample. '
                  'Ablation uses test days only. Indoor comfort is physics-modelled; DHT is live/calibration only.'}
    kpis = metrics.compute_kpis(sched, sens, prov, fi)
    kpis['days_at_cut_ceiling'] = int(sched.groupby('date')['ceiling_hit'].first().sum())
    kpis['indoor_profile_label'] = params.get('profile_label', 'assumed')

    out = sched.copy()
    out['timestamp'] = out['timestamp'].astype(str)
    out.to_csv('data/processed/optimized_schedule.csv', index=False)
    sens.to_csv('results/sensitivity.csv', index=False)
    _json('results/kpis.json', kpis)
    _json('results/ablation.json', abl)
    demo = exports.select_demo_day(fdf)
    exports.write_all(demo)  # demo_day.json, replay_table.h, magnus.h, HARDWARE_CONTRACT.md
    open('results/claims.md', 'w').write(report.claims_md(kpis, abl))
    open('docs/EVIDENCE.md', 'w').write(evidence_md())
    report.update_readme(kpis)

    h = kpis['headline']
    print(f"\n=== SmartCool KPIs ({prov['data_source']}, {kpis['scope']['days']} days) ===")
    print(f"  peak reduction        : {h['peak_reduction_pct']}%")
    print(f"  energy (closed-loop)  : {h['energy_change_pct_closed_loop']}%   open-loop {kpis['energy_change_open_loop_pct']}%")
    print(f"  comfort (modelled)    : {h['comfort_pct_indoor_modelled']}%   max indoor {kpis['max_indoor_c']} C")
    print(f"  indoor profile        : {kpis['indoor_profile_label']}")
    print(f"  modes                 : {kpis['mode_counts']}")
    print(f"  forecast              : {abl['verdict']}")
    print(f"  demo day              : {demo['date']} scenario={demo['scenario']['name'] if demo['scenario'] else 'none'} "
          f"loop {demo['loop_seconds']} s")
    print(f"  hardware artefacts    : firmware/HARDWARE_CONTRACT.md, firmware/replay_table.h, results/demo_day.json")
    return sched, kpis


if __name__ == '__main__':
    run_pipeline()