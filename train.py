"""Train and save model artefacts. Run explicitly: `python train.py`. pipeline.py never overwrites them."""
from logic.features import engineer_features
from logic.load_data import load_data
from logic.model import train_ac_predictor, train_peak_detector

if __name__ == '__main__':
    df = engineer_features(load_data())
    train_ac_predictor(df)      # legacy lag-based regressor (kept for the legacy tests)
    train_peak_detector(df)     # legacy peak classifier (nowcasts from lagged load; see I4)
    # The day-ahead forecaster (logic/forecast.py) is deterministic (seeded) and cheap, so it is
    # retrained in memory by pipeline.py rather than stored.
