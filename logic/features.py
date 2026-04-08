import pandas as pd

def engineer_features(df):
    df = df.copy()

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([4, 5]).astype(int)  # Fri/Sat in Saudi Arabia

    df['ac_lag_1h']      = df['ac_kwh'].shift(1)   # what AC was 1 hour ago
    df['ac_lag_2h']      = df['ac_kwh'].shift(2)
    df['ac_lag_24h']     = df['ac_kwh'].shift(24)  # same hour yesterday

    df['ac_roll_3h']     = df['ac_kwh'].rolling(3).mean()
    df['ac_roll_6h']     = df['ac_kwh'].rolling(6).mean()

    df['temp_humidity_index'] = df['temp'] + 0.33 * df['humidity'] - 4  # heat discomfort index
    df['temp_lag_1h']    = df['temp'].shift(1)
    df['humidity_lag_1h'] = df['humidity'].shift(1)

    df['pre_peak'] = df['hour'].isin([10, 11]).astype(int)

    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns():
    return [
        'hour', 'day_of_week', 'is_weekend',
        'temp', 'humidity', 'dew_point', 'solar',
        'temp_lag_1h', 'humidity_lag_1h',
        'temp_humidity_index',
        'ac_lag_1h', 'ac_lag_2h', 'ac_lag_24h',
        'ac_roll_3h', 'ac_roll_6h',
        'pre_peak'
    ]