"""Train and save model artefacts. Run explicitly: `python train.py`. pipeline.py never overwrites them.

Legacy pickles only (Analytics / old tests). The active day-ahead RF lives in logic/forecast.py,
is outdoor-weather-only, and is retrained in memory by pipeline.py — never from indoor or DHT logs.
"""
from logic.features import engineer_features
from logic.load_data import load_data
from logic.model import train_ac_predictor, train_peak_detector

if __name__ == '__main__':
    df = engineer_features(load_data())
    train_ac_predictor(df)      # legacy lag-based regressor (kept for the legacy tests)
    train_peak_detector(df)     # legacy peak classifier (nowcasts from lagged load; see I4)
    # Day-ahead forecaster (logic/forecast.py): outdoor FEATURES only; seeded; in-memory in pipeline.py.
