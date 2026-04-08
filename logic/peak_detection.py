import pandas as pd
import joblib
from logic.features import get_feature_columns

def predict_peaks(df, lookahead_hours=2):
    model = joblib.load('logic/peak_detector.pkl')
    features = [f for f in get_feature_columns() if f != 'hour']  # must match training

    df = df.copy()
    df['predicted_peak'] = model.predict(df[features])
    df['predicted_peak_proba'] = model.predict_proba(df[features])[:, 1]

    df['pre_cool_window'] = 0
    for i in range(1, lookahead_hours + 1):
        df['pre_cool_window'] = df['pre_cool_window'] | df['predicted_peak'].shift(-i).fillna(0).astype(int)

    return df

def get_peak_summary(df):
   
    total_hours = len(df)
    actual_peaks = df['is_peak'].sum()
    predicted_peaks = df['predicted_peak'].sum()
    correct = ((df['predicted_peak'] == 1) & (df['is_peak'] == 1)).sum()
    missed = ((df['predicted_peak'] == 0) & (df['is_peak'] == 1)).sum()
    false_alarms = ((df['predicted_peak'] == 1) & (df['is_peak'] == 0)).sum()

    print("=== Peak Detection Summary ===")
    print(f"Total hours analyzed  : {total_hours}")
    print(f"Actual peak hours     : {actual_peaks}")
    print(f"Predicted peak hours  : {predicted_peaks}")
    print(f"Correctly detected    : {correct}")
    print(f"Missed peaks          : {missed}  ← Hamza's pre-cool won't trigger here")
    print(f"False alarms          : {false_alarms}  ← Pre-cool triggers unnecessarily")
    print(f"Detection rate        : {round(correct / actual_peaks * 100, 1)}%")


def export_peak_schedule(df, output_path='logic/peak_schedule.csv'):
   
    cols = ['timestamp', 'hour', 'is_peak', 'predicted_peak', 'predicted_peak_proba', 'pre_cool_window']
    schedule = df[cols].copy()
    schedule.to_csv(output_path, index=False)
    print(f"Peak schedule saved to: {output_path}")
    return schedule