import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SCALED_COLS = ["age", "education-num", "capital-gain", "capital-loss"]
LOG_COLS = ["capital-gain", "capital-loss"]
MODEL_FILES = {
    "AdaBoost ": "reduced_adaboost_model_mina.pkl",
    "Gradient Boosting ": "reduced_GBmodel_heba.pkl",
}

data = pd.read_csv("census.csv")
data[LOG_COLS] = np.log(data[LOG_COLS] + 1)          

# MinMaxScaler works column by column, so fitting on these 4 columns gives
scaler = MinMaxScaler()
scaler.fit(data[SCALED_COLS])
joblib.dump(scaler, "scaler.pkl")

model_features = {}
for name, path in MODEL_FILES.items():
    if os.path.exists(path):
        m = joblib.load(path)
        model_features[name] = list(getattr(m, "feature_names_in_", []))

schema = {
    # HTML form `name` attribute  ->  dataset column
    "form_fields": {
        "age": "age",
        "education_num": "education-num",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "marital_status": "marital-status_Married-civ-spouse",
    },
    "ranges": {
        "age": [17, 100],
        "education_num": [1, 16],
        "capital_gain": [0, 99999],
        "capital_loss": [0, 5000],
    },
    "marital_status_options": ["Single", "Married"],
    "scaled_columns": SCALED_COLS,
    "scaler_min": dict(zip(SCALED_COLS, scaler.data_min_.round(4).tolist())),
    "scaler_max": dict(zip(SCALED_COLS, scaler.data_max_.round(4).tolist())),
    "model_features": model_features,
}
with open("schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("Saved scaler.pkl and schema.json")
for name, feats in model_features.items():
    print(f"{name} expects (in this order): {feats}")