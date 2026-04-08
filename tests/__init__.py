import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.load_data import load_data
from logic.features import engineer_features, get_feature_columns
from logic.peak_detection import predict_peaks, get_peak_summary

def passed(test_name):
    print(f"  PASSED: {test_name}")

def failed(test_name, reason):
    print(f"  FAILED: {test_name} — {reason}")

def test_data_loads():
    df = load_data('HEMS_Sample_Dataset.xlsx')
    if df is None or len(df) == 0:
        failed("Data loads", "DataFrame is empty")
    else:
        passed("Data loads")
    return df

def test_expected_rows(df):
    if len(df) == 720:
        passed("Row count is 720 (30 days × 24 hours)")
    else:
        failed("Row count", f"Expected 720, got {len(df)}")

def test_no_nulls(df):
    nulls = df.isnull().sum().sum()
    if nulls == 0:
        passed("No null values in raw data")
    else:
        failed("No null values", f"Found {nulls} nulls")

def test_columns_exist(df):
    required = ['timestamp', 'temp', 'humidity', 'dew_point', 'ac_kwh', 'total_kwh', 'is_peak']
    missing = [c for c in required if c not in df.columns]
    if not missing:
        passed("All required columns present")
    else:
        failed("Required columns", f"Missing: {missing}")

def test_timestamp_sorted(df):
    if df['timestamp'].is_monotonic_increasing:
        passed("Timestamps are sorted ascending")
    else:
        failed("Timestamps sorted", "Timestamps are not in order")

def test_peak_hours_correct(df):
    # Peak hours should only be 12–18 based on dataset
    peak_df = df[df['is_peak'] == 1]
    non_peak_hours = peak_df[~peak_df['hour'].isin(range(12, 19))]['hour'].unique()
    if len(non_peak_hours) == 0:
        passed("Peak hours are only within 12pm–6pm")
    else:
        failed("Peak hours range", f"Unexpected peak hours found: {non_peak_hours}")

def test_temperature_range(df):
    if df['temp'].between(0, 60).all():
        passed("Temperature values in realistic range (0–60°C)")
    else:
        failed("Temperature range", f"Min: {df['temp'].min()}, Max: {df['temp'].max()}")

def test_humidity_range(df):
    if df['humidity'].between(0, 100).all():
        passed("Humidity values in valid range (0–100%)")
    else:
        failed("Humidity range", f"Min: {df['humidity'].min()}, Max: {df['humidity'].max()}")

def test_ac_kwh_positive(df):
    if (df['ac_kwh'] >= 0).all():
        passed("AC consumption values are all non-negative")
    else:
        failed("AC kWh positive", f"Found {(df['ac_kwh'] < 0).sum()} negative values")

def test_features_created(df_raw):
    df = engineer_features(df_raw)
    expected = get_feature_columns()
    missing = [f for f in expected if f not in df.columns]
    if not missing:
        passed("All engineered features created")
    else:
        failed("Feature creation", f"Missing features: {missing}")
    return df

def test_no_nulls_after_features(df):
    nulls = df[get_feature_columns()].isnull().sum().sum()
    if nulls == 0:
        passed("No nulls in feature columns after engineering")
    else:
        failed("Nulls after feature engineering", f"Found {nulls} nulls")

def test_row_drop_reasonable(df_raw, df_featured):
    dropped = len(df_raw) - len(df_featured)
    if dropped <= 30:
        passed(f"Row drop after feature engineering is acceptable ({dropped} rows dropped)")
    else:
        failed("Row drop", f"Too many rows dropped: {dropped}")

def test_lag_features_shift_correctly(df):
    sample = df.iloc[5]
    expected = df.iloc[4]['ac_kwh']
    actual = sample['ac_lag_1h']
    if abs(actual - expected) < 0.0001:
        passed("Lag feature (ac_lag_1h) shifts correctly")
    else:
        failed("Lag feature shift", f"Expected {expected}, got {actual}")

def test_weekend_flag(df):
    weekend_hours = df[df['is_weekend'] == 1]['day_of_week'].unique()
    non_weekend = [d for d in weekend_hours if d not in [4, 5]]
    if not non_weekend:
        passed("Weekend flag correctly marks Friday(4) and Saturday(5)")
    else:
        failed("Weekend flag", f"Unexpected days flagged as weekend: {non_weekend}")

def test_pre_peak_flag(df):
    pre_peak_hours = df[df['pre_peak'] == 1]['hour'].unique()
    unexpected = [h for h in pre_peak_hours if h not in [10, 11]]
    if not unexpected:
        passed("Pre-peak flag correctly marks hours 10 and 11")
    else:
        failed("Pre-peak flag", f"Unexpected hours flagged: {unexpected}")

def test_models_saved():
    if os.path.exists('logic/ac_predictor.pkl'):
        passed("ac_predictor.pkl exists")
    else:
        failed("ac_predictor.pkl", "File not found — run pipeline.py first")

    if os.path.exists('logic/peak_detector.pkl'):
        passed("peak_detector.pkl exists")
    else:
        failed("peak_detector.pkl", "File not found — run pipeline.py first")

def test_models_loadable():
    try:
        joblib.load('logic/ac_predictor.pkl')
        joblib.load('logic/peak_detector.pkl')
        passed("Both models load without errors")
    except Exception as e:
        failed("Model loading", str(e))

def test_ac_model_prediction_shape(df):
    try:
        model = joblib.load('logic/ac_predictor.pkl')
        features = get_feature_columns()
        preds = model.predict(df[features])
        if len(preds) == len(df):
            passed("AC model output shape matches input rows")
        else:
            failed("AC model output shape", f"Expected {len(df)}, got {len(preds)}")
    except Exception as e:
        failed("AC model prediction", str(e))

def test_ac_predictions_positive(df):
    try:
        model = joblib.load('logic/ac_predictor.pkl')
        features = get_feature_columns()
        preds = model.predict(df[features])
        if (preds >= 0).all():
            passed("AC model predicts only non-negative values")
        else:
            failed("AC predictions positive", f"{(preds < 0).sum()} negative predictions")
    except Exception as e:
        failed("AC predictions positive", str(e))
        
def test_peak_model_binary_output(df):
    try:
        model = joblib.load('logic/peak_detector.pkl')
        features = [f for f in get_feature_columns() if f != 'hour']  # match training
        preds = model.predict(df[features])
        unique_vals = set(preds)
        if unique_vals.issubset({0, 1}):
            passed("Peak detector outputs only 0 or 1")
        else:
            failed("Peak detector binary", f"Unexpected values: {unique_vals}")
    except Exception as e:
        failed("Peak detector binary output", str(e))

def test_peak_detection_rate(df):
    try:
        df_peaks = predict_peaks(df, lookahead_hours=2)
        actual = df_peaks['is_peak'].sum()
        detected = ((df_peaks['predicted_peak'] == 1) & (df_peaks['is_peak'] == 1)).sum()
        rate = detected / actual * 100
        if rate >= 80:
            passed(f"Peak detection rate is acceptable ({rate:.1f}% of actual peaks caught)")
        else:
            failed("Peak detection rate", f"Only {rate:.1f}% detected — model needs improvement")
    except Exception as e:
        failed("Peak detection rate", str(e))

def test_pre_cool_window_lookahead(df):
    try:
        df_peaks = predict_peaks(df, lookahead_hours=2)
        # Every pre_cool_window=1 should have a predicted_peak=1 within next 2 hours
        errors = 0
        for i in range(len(df_peaks) - 2):
            if df_peaks.iloc[i]['pre_cool_window'] == 1:
                next_2 = df_peaks.iloc[i+1:i+3]['predicted_peak'].max()
                if next_2 == 0:
                    errors += 1
        if errors == 0:
            passed("Pre-cool window correctly precedes predicted peaks")
        else:
            failed("Pre-cool window lookahead", f"{errors} windows not followed by predicted peak")
    except Exception as e:
        failed("Pre-cool window", str(e))


if __name__ == '__main__':
    print("\n========================================")
    print("   HEMS MODEL VALIDATION")
    print("========================================")

    print("\n [1] DATA LOADER TESTS")
    df_raw = test_data_loads()
    test_expected_rows(df_raw)
    test_no_nulls(df_raw)
    test_columns_exist(df_raw)
    test_timestamp_sorted(df_raw)
    test_peak_hours_correct(df_raw)
    test_temperature_range(df_raw)
    test_humidity_range(df_raw)
    test_ac_kwh_positive(df_raw)

    print("\n [2] FEATURE ENGINEERING TESTS")
    df_featured = test_features_created(df_raw)
    test_no_nulls_after_features(df_featured)
    test_row_drop_reasonable(df_raw, df_featured)
    test_lag_features_shift_correctly(df_featured)
    test_weekend_flag(df_featured)
    test_pre_peak_flag(df_featured)

    print("\n [3] MODEL TESTS")
    test_models_saved()
    test_models_loadable()
    test_ac_model_prediction_shape(df_featured)
    test_ac_predictions_positive(df_featured)
    test_peak_model_binary_output(df_featured)
    test_peak_detection_rate(df_featured)
    test_pre_cool_window_lookahead(df_featured)

    print("\n========================================")
    print("   DONE — fix any errors before handoff")
    print("========================================\n")