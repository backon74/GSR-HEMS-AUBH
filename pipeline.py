from logic.load_data import load_data
from logic.features import engineer_features
from logic.model import train_ac_predictor, train_peak_detector
from logic.peak_detection import predict_peaks, get_peak_summary, export_peak_schedule
from logic.control_logic import apply_control_logic, get_control_summary
from logic.scheduler import build_full_schedule

def run_pipeline():
    df_raw = load_data('HEMS_Sample_Dataset.xlsx')
    df = engineer_features(df_raw)

    ac_model   = train_ac_predictor(df)
    peak_model = train_peak_detector(df)

    # Peak detection (Hana)
    df = predict_peaks(df, lookahead_hours=2)
    get_peak_summary(df)
    export_peak_schedule(df)

    # Control logic + scheduling (Hamza)
    df = apply_control_logic(df)
    kpis = get_control_summary(df)

    print("\n=== Control KPIs ===")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    build_full_schedule(df)

    return df, ac_model, peak_model

if __name__ == '__main__':
    run_pipeline()