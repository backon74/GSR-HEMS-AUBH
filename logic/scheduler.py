"""
logic/scheduler.py
===================
Hamza's optimized A/C scheduling system.

Builds a human-readable hourly schedule that tells the smart thermostat
what to do each hour: mode, setpoint adjustment, expected load.

Depends on control_logic.apply_control_logic() having already been run.
"""

import pandas as pd


def build_daily_schedule(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Returns the optimized hourly schedule for a single day.

    Parameters
    ----------
    df   : DataFrame after apply_control_logic()
    date : string e.g. '2025-07-02'
    """
    mask = df['timestamp'].dt.date.astype(str) == date
    day  = df[mask].copy().sort_values('hour').reset_index(drop=True)

    schedule = day[[
        'hour', 'temp', 'humidity', 'dew_point',
        'ac_kwh', 'optimized_ac_kwh', 'ac_saved_kwh',
        'control_mode', 'setpoint_adj',
        'comfort_score', 'predicted_peak', 'pre_cool_window'
    ]].copy()

    schedule['action'] = schedule.apply(_describe_action, axis=1)
    return schedule


def build_full_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the complete optimized schedule for all days in the dataset.
    Saves to data/processed/optimized_schedule.csv automatically.
    """
    out = df[[
        'timestamp', 'date', 'hour', 'building_id',
        'control_mode', 'setpoint_adj',
        'ac_kwh', 'optimized_ac_kwh', 'ac_saved_kwh',
        'comfort_score', 'predicted_peak', 'pre_cool_window',
        'predicted_peak_proba'
    ]].copy().sort_values(['timestamp']).reset_index(drop=True)

    out['action'] = out.apply(_describe_action, axis=1)

    out.to_csv('data/processed/optimized_schedule.csv', index=False)
    print(f"Optimized schedule saved → data/processed/optimized_schedule.csv  ({len(out)} rows)")
    return out


def _describe_action(row) -> str:
    mode = row['control_mode']
    adj  = row['setpoint_adj']
    if mode == 'pre_cool':
        return f"Pre-cool: lower setpoint {abs(adj):.1f}°C (peak in ≤2h)"
    elif mode == 'peak_reduce':
        return f"Peak reduce: raise setpoint {adj:.1f}°C (cut 25% load)"
    elif mode == 'comfort_override':
        return "Comfort override: hold cooling (humidity too high)"
    else:
        if adj > 0:
            return f"Normal: raise setpoint {adj:.1f}°C (low humidity)"
        elif adj < 0:
            return f"Normal: lower setpoint {abs(adj):.1f}°C (high humidity/heat)"
        return "Normal: hold current setpoint"