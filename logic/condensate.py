
import pandas as pd
import numpy as np

BASE_LITRES_PER_KWH = 0.55
HUMIDITY_SCALE = 1.6
TEMP_OFFSET = 35.0
TEMP_SCALE = 0.018


def estimate_condensate(ac_kwh: float, humidity_pct: float, temp: float) -> float:
    #Estimate condensate water production in litres per hour.

    if ac_kwh <= 0:
        return 0.0

    humidity_factor = (humidity_pct / 100.0) * HUMIDITY_SCALE
    temp_factor     = 1.0 + (temp - TEMP_OFFSET) * TEMP_SCALE
    temp_factor     = max(0.5, temp_factor)   # floor at 0.5

    litres = ac_kwh * BASE_LITRES_PER_KWH * humidity_factor * temp_factor
    return round(litres, 4)


def add_condensate_columns(df: pd.DataFrame) -> pd.DataFrame:
    #Adds condensate estimates for both baseline and optimized A/C loads.

    df = df.copy()

    df['condensate_baseline_L'] = df.apply(
        lambda r: estimate_condensate(r['ac_kwh'], r['humidity'], r['dew_point']), axis=1
    )
    df['condensate_optimized_L'] = df.apply(
        lambda r: estimate_condensate(r['optimized_ac_kwh'], r['humidity'], r['dew_point']), axis=1
    )
    df['condensate_total_L'] = df['condensate_optimized_L']

    return df


def get_condensate_summary(df: pd.DataFrame) -> dict:
    #condensate estimate summary
    if 'condensate_optimized_L' not in df.columns:
        df = add_condensate_columns(df)

    total_L   = df['condensate_optimized_L'].sum()
    daily_avg = total_L / df['timestamp'].dt.date.nunique()
    yearly_est = daily_avg * 365

    return {
        'monthly_litres':      round(total_L, 1),
        'daily_avg_litres':    round(daily_avg, 1),
        'yearly_est_litres':   round(yearly_est, 0),
        'yearly_est_m3':       round(yearly_est / 1000, 2),
    }