import pandas as pd

def load_data(filepath='HEMS_Sample_Dataset.xlsx'):
    df = pd.read_excel(filepath, sheet_name='Hourly_Data')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    df = df.rename(columns={
        'temperature_c': 'temp',
        'humidity_pct': 'humidity',
        'dew_point_c': 'dew_point',
        'ac_consumption_kwh': 'ac_kwh',
        'total_consumption_kwh': 'total_kwh',
        'solar_irradiance_wm2': 'solar',
        'is_peak_hour': 'is_peak'
    })
    print(f"Loaded {len(df)} rows from {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    return df