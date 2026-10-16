import json
import os

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "schema.json")) as f:
    SCHEMA = json.load(f)
SCALER = joblib.load(os.path.join(BASE, "scaler.pkl"))

FORM_FIELDS = SCHEMA["form_fields"]
RANGES = SCHEMA["ranges"]
MARITAL_OPTIONS = SCHEMA["marital_status_options"]
SCALED_COLS = SCHEMA["scaled_columns"]
MARITAL_COL = FORM_FIELDS["marital_status"]

# display name -> file name (repo root)
MODEL_FILES = {
    "AdaBoost": "reduced_adaboost_model_mina.pkl",
    "Gradient Boosting": "reduced_GBmodel_heba.pkl",
}
MODELS = {}
for _name, _file in MODEL_FILES.items():
    _path = os.path.join(BASE, _file)
    if os.path.exists(_path):
        MODELS[_name] = joblib.load(_path)
if not MODELS:
    raise RuntimeError("No model .pkl found next to preprocess.py")


def validate(form) -> dict:
    """Check the 5 fields. Returns clean values or raises ValueError (readable message)."""
    errors, clean = [], {}
    for field in FORM_FIELDS:
        value = str(form.get(field, "")).strip()
        if value == "":
            errors.append(f"'{field}' is required.")
            continue

        if field == "marital_status":
            if value not in MARITAL_OPTIONS:
                errors.append(f"'{value}' is not an allowed value for 'marital_status'.")
            else:
                clean[field] = value
            continue

        try:
            num = float(value)
        except ValueError:
            errors.append(f"'{field}' must be a number.")
            continue
        low, high = RANGES[field]
        if not (low <= num <= high):
            errors.append(f"'{field}' must be between {low} and {high}.")
            continue
        if field in ("age", "education_num") and num != int(num):
            errors.append(f"'{field}' must be a whole number.")
            continue
        clean[field] = num

    if errors:
        raise ValueError(" ".join(errors))
    return clean


def preprocess(form) -> pd.DataFrame:
    """One-row DataFrame with all 5 processed columns (order is fixed later per model)."""
    c = validate(form)
    df = pd.DataFrame([{
        "age": c["age"],
        "education-num": c["education_num"],
        "capital-gain": np.log(c["capital_gain"] + 1),
        "capital-loss": np.log(c["capital_loss"] + 1),
    }])
    df[SCALED_COLS] = SCALER.transform(df[SCALED_COLS])
    df[SCALED_COLS] = df[SCALED_COLS].clip(0, 1)      # e.g. age 100 > max age seen in training
    df[MARITAL_COL] = 1.0 if c["marital_status"] == "Married" else 0.0
    return df


def _norm(name: str) -> str:
    """census.csv values have a leading space, so the models' column is named
    'marital-status_ Married-civ-spouse'. Ignore that space when matching names."""
    return name.replace("_ ", "_").strip()


def features_for(model, df: pd.DataFrame) -> pd.DataFrame:
    """Select/reorder columns exactly as this model was trained (using the model's own names)."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise RuntimeError("Model has no feature_names_in_; it was not trained on a DataFrame.")
    by_norm = {_norm(c): c for c in df.columns}
    missing = [n for n in names if _norm(n) not in by_norm]
    if missing:
        raise RuntimeError(f"Model needs columns the form does not provide: {missing}")
    out = df[[by_norm[_norm(n)] for n in names]].copy()
    out.columns = list(names)                       # exact names the model remembers
    return out


def predict_all(form) -> list:
    """Run every loaded model. Raises ValueError for bad input."""
    df = preprocess(form)
    results = []
    for name, model in MODELS.items():
        X = features_for(model, df)
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1]) * 100
        results.append({
            "model": name,
            "prediction": pred,
            "label": "Likely donor (income > $50K)" if pred == 1 else "Unlikely donor (income <= $50K)",
            "probability": round(proba, 1),
        })
    return results