
import pandas as pd

TARIFF_OFFPEAK_SAR  = 0.18   # SAR per kWh — standard residential
TARIFF_PEAK_SAR     = 0.30   # SAR per kWh — peak TOU (projected DR rate)

EP_RESIDENTIAL_HOMES    = 850_000   # Eastern Province residential units (~2023 census)
AVG_HOMES_PER_HOOD      = 500       # typical neighbourhood size
CO2_KG_PER_KWH          = 0.64      # Saudi grid emission factor (IEA 2023)


def _hourly_tariff(is_peak: int) -> float:
    return TARIFF_PEAK_SAR if is_peak else TARIFF_OFFPEAK_SAR

def add_cost_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['tariff_sar']        = df['predicted_peak'].apply(_hourly_tariff)
    df['cost_baseline_sar'] = (df['ac_kwh']           * df['tariff_sar']).round(4)
    df['cost_optimized_sar']= (df['optimized_ac_kwh'] * df['tariff_sar']).round(4)
    df['cost_saved_sar']    = (df['cost_baseline_sar'] - df['cost_optimized_sar']).clip(lower=0).round(4)
    return df


def get_cost_summary(df: pd.DataFrame) -> dict:
    
    if 'cost_saved_sar' not in df.columns:
        df = add_cost_columns(df)

    baseline_cost  = df['cost_baseline_sar'].sum()
    optimized_cost = df['cost_optimized_sar'].sum()
    saved_sar      = df['cost_saved_sar'].sum()
    n_days         = df['timestamp'].dt.date.nunique()

    return {
        'total_baseline_sar':  round(baseline_cost, 2),
        'total_optimized_sar': round(optimized_cost, 2),
        'total_saved_sar':     round(saved_sar, 2),
        'daily_avg_saved_sar': round(saved_sar / n_days, 2),
        'monthly_saved_sar':   round(saved_sar, 2),
        'yearly_est_saved_sar':round(saved_sar / n_days * 365, 2),
    }


def get_scaling_summary(df: pd.DataFrame) -> dict:
    if 'cost_saved_sar' not in df.columns:
        df = add_cost_columns(df)

    n_days         = df['timestamp'].dt.date.nunique()
    saved_kwh_day  = (df['ac_kwh'] - df['optimized_ac_kwh']).clip(lower=0).sum() / n_days
    saved_sar_day  = df['cost_saved_sar'].sum() / n_days

    peak_df        = df[df['predicted_peak'] == 1]
    peak_baseline  = peak_df['ac_kwh'].mean()
    peak_optimized = peak_df['optimized_ac_kwh'].mean()
    peak_kw_saved_per_home = (peak_baseline - peak_optimized)  # kW average during peak

    hood_kwh_day   = saved_kwh_day  * AVG_HOMES_PER_HOOD
    hood_sar_day   = saved_sar_day  * AVG_HOMES_PER_HOOD
    hood_mw_peak   = peak_kw_saved_per_home * AVG_HOMES_PER_HOOD / 1000

    ep_kwh_year    = saved_kwh_day  * EP_RESIDENTIAL_HOMES * 365
    ep_sar_year    = saved_sar_day  * EP_RESIDENTIAL_HOMES * 365
    ep_mw_peak     = peak_kw_saved_per_home * EP_RESIDENTIAL_HOMES / 1000
    ep_co2_year_t  = ep_kwh_year    * CO2_KG_PER_KWH / 1000   # tonnes

    return {
        'per_home_kwh_saved_day':   round(saved_kwh_day, 2),
        'per_home_sar_saved_day':   round(saved_sar_day, 2),
        'per_home_sar_saved_year':  round(saved_sar_day * 365, 0),

        'hood_kwh_saved_day':       round(hood_kwh_day, 1),
        'hood_sar_saved_day':       round(hood_sar_day, 1),
        'hood_mw_peak_reduced':     round(hood_mw_peak, 3),

        'ep_gwh_saved_year':        round(ep_kwh_year / 1_000_000, 1),
        'ep_sar_saved_year_B':      round(ep_sar_year / 1_000_000_000, 2),   # billions
        'ep_mw_peak_reduced':       round(ep_mw_peak, 0),
        'ep_co2_saved_tonnes_year': round(ep_co2_year_t, 0),
    }