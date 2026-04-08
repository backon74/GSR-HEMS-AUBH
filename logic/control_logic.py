"""
logic/control_logic.py
========================
Hamza's smart A/C control logic.

Receives the DataFrame AFTER Hana's full pipeline has run, meaning these
columns are already present and ready to use:
    predicted_peak         (0/1)  — Hana's RandomForestClassifier output
    predicted_peak_proba   (float) — confidence score
    pre_cool_window        (0/1)  — 1 if a peak is coming within 2 hours
    temp, humidity, dew_point, ac_kwh, hour, ...

Adds these new columns:
    control_mode       — 'normal' | 'pre_cool' | 'peak_reduce' | 'comfort_override'
    setpoint_adj       — °C to raise(+) or lower(-) the thermostat setpoint
    optimized_ac_kwh   — estimated A/C load after control is applied
    comfort_score      — 0–1 score (1 = very comfortable)
    safe_to_reduce     — bool: is it thermally safe to cut cooling right now
    ac_saved_kwh       — baseline minus optimized (clipped at 0)
"""

import pandas as pd
import numpy as np

# ── Comfort thresholds (coordinate with Zainab) ───────────────────────────────
DEW_POINT_UNCOMFORTABLE = 24.0   # °C — above this, humidity feels oppressive
TEMP_EXTREME            = 47.0   # °C — above this, never reduce cooling

# ── Control parameters ────────────────────────────────────────────────────────
PEAK_REDUCE_FRACTION  = 0.25     # Cut A/C by 25% during peak (pre-cool already ran)
PRE_COOL_BOOST_KWH    = 0.15     # Extra kWh added during pre-cool window
MIN_AC_KWH            = 0.10     # Never go below this (keeps airflow alive)
# ─────────────────────────────────────────────────────────────────────────────


def _comfort_score(temp: float, dew_point: float) -> float:
    """0 = very uncomfortable, 1 = very comfortable."""
    temp_stress = max(0.0, (temp - 35.0) / 20.0)
    dew_stress  = max(0.0, (dew_point - DEW_POINT_UNCOMFORTABLE) / 10.0)
    return round(max(0.0, 1.0 - (0.6 * temp_stress + 0.4 * dew_stress)), 3)


def _safe_to_reduce(temp: float, dew_point: float) -> bool:
    """Return True if it is safe to reduce cooling."""
    if dew_point >= DEW_POINT_UNCOMFORTABLE:
        return False
    if temp >= TEMP_EXTREME:
        return False
    return True


def apply_control_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Call this after Hana's predict_peaks().

    Parameters
    ----------
    df : DataFrame returned by predict_peaks() — contains predicted_peak,
         predicted_peak_proba, pre_cool_window, temp, humidity, dew_point,
         ac_kwh, hour, etc.

    Returns
    -------
    df with six new columns added (see module docstring).
    """
    df = df.copy()

    control_modes  = []
    setpoint_adjs  = []
    optimized_ac   = []
    comfort_scores = []
    safe_flags     = []

    for _, row in df.iterrows():
        temp      = row['temp']
        dew       = row['dew_point']
        base_ac   = row['ac_kwh']
        is_peak   = row['predicted_peak'] == 1
        pre_cool  = row['pre_cool_window'] == 1

        score = _comfort_score(temp, dew)
        safe  = _safe_to_reduce(temp, dew)

        if is_peak:
            if safe:
                # ── PEAK: reduce load — pre-cooling already built thermal buffer
                mode    = 'peak_reduce'
                opt_ac  = max(MIN_AC_KWH, base_ac * (1 - PEAK_REDUCE_FRACTION))
                adj     = +1.0   # raise setpoint → less cooling
            else:
                # ── Humidity/heat too high — comfort override, hold cooling
                mode    = 'comfort_override'
                opt_ac  = base_ac
                adj     = -0.5

        elif pre_cool:
            # ── PRE-COOL: ramp up cooling now to build thermal mass
            mode    = 'pre_cool'
            opt_ac  = base_ac + PRE_COOL_BOOST_KWH
            adj     = -1.0   # lower setpoint → more cooling

        else:
            # ── NORMAL: minor humidity-aware trim
            mode = 'normal'
            if dew < 16.0 and temp < 40.0:
                adj    = +1.5    # low humidity → slightly reduce cooling
                opt_ac = max(MIN_AC_KWH, base_ac - 0.05)
            elif dew >= DEW_POINT_UNCOMFORTABLE or temp >= 46.0:
                adj    = -1.0   # very humid/hot → keep cooling strong
                opt_ac = base_ac
            else:
                adj    = 0.0
                opt_ac = base_ac

        control_modes.append(mode)
        setpoint_adjs.append(adj)
        optimized_ac.append(round(opt_ac, 4))
        comfort_scores.append(score)
        safe_flags.append(safe)

    df['control_mode']     = control_modes
    df['setpoint_adj']     = setpoint_adjs
    df['optimized_ac_kwh'] = optimized_ac
    df['comfort_score']    = comfort_scores
    df['safe_to_reduce']   = safe_flags
    df['ac_saved_kwh']     = (df['ac_kwh'] - df['optimized_ac_kwh']).clip(lower=0).round(4)

    return df


def get_control_summary(df: pd.DataFrame) -> dict:
    """
    Returns a dict of KPIs for Noor's evaluation module.

    Keys
    ----
    peak_reduction_pct    : % reduction in A/C load during peak hours
    energy_savings_pct    : % total A/C energy saved across all hours
    comfort_pct           : % hours where comfort_score >= 0.6
    total_ac_baseline_kwh : sum of original ac_kwh
    total_ac_optimized_kwh: sum of optimized_ac_kwh
    total_saved_kwh       : baseline minus optimized
    peak_hours_reduced    : count of peak hours where load was actually cut
    mode_counts           : dict of {mode: hour_count}
    """
    baseline  = df['ac_kwh'].sum()
    optimized = df['optimized_ac_kwh'].sum()
    saved     = baseline - optimized

    peak_df       = df[df['predicted_peak'] == 1]
    peak_baseline = peak_df['ac_kwh'].sum()
    peak_optimized= peak_df['optimized_ac_kwh'].sum()
    peak_reduction = (
        (peak_baseline - peak_optimized) / peak_baseline * 100
        if peak_baseline > 0 else 0.0
    )

    return {
        'peak_reduction_pct':     round(peak_reduction, 1),
        'energy_savings_pct':     round(saved / baseline * 100, 1),
        'comfort_pct':            round((df['comfort_score'] >= 0.6).mean() * 100, 1),
        'total_ac_baseline_kwh':  round(baseline, 2),
        'total_ac_optimized_kwh': round(optimized, 2),
        'total_saved_kwh':        round(saved, 2),
        'peak_hours_reduced':     int((df['control_mode'] == 'peak_reduce').sum()),
        'mode_counts':            df['control_mode'].value_counts().to_dict(),
    }