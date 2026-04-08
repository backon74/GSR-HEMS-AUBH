import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report

from logic.features import get_feature_columns

def train_ac_predictor(df):
    features = get_feature_columns()
    X = df[features]
    y = df['ac_kwh']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"[AC Predictor] MAE: {mae:.4f} kWh")

    joblib.dump(model, 'logic/ac_predictor.pkl')
    print("Saved: logic/ac_predictor.pkl")
    return model

def train_peak_detector(df):
    features = [f for f in get_feature_columns() if f != 'hour']
    
    X = df[features]
    y = df['is_peak']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("[Peak Detector] Classification Report:")
    print(classification_report(y_test, preds))

    joblib.dump(model, 'logic/peak_detector.pkl')
    print("Saved: logic/peak_detector.pkl")
    return model

def load_models():
    ac_model = joblib.load('logic/ac_predictor.pkl')
    peak_model = joblib.load('logic/peak_detector.pkl')
    return ac_model, peak_model