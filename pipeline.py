from logic.load_data import load_data
from logic.features import engineer_features
from logic.model import train_ac_predictor, train_peak_detector

def run_pipeline():
    df_raw = load_data('HEMS_Sample_Dataset.xlsx')

    df = engineer_features(df_raw)
    print(f"Features ready. Shape: {df.shape}")

    ac_model = train_ac_predictor(df)
    peak_model = train_peak_detector(df)

    return df, ac_model, peak_model


if __name__ == '__main__':
    run_pipeline()