"verify notebook.py is as expected, and that the app gives the same predictions as the notebook"

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")          
import preprocess as P                      

N_ROWS = 2000                               # how many test rows to compare

# ---------- A) notebook pipeline ----------
data = pd.read_csv("census.csv")
income = data["income"].map({"<=50K": 0, ">50K": 1})
features = data.drop("income", axis=1)
features[["capital-gain", "capital-loss"]] = np.log(features[["capital-gain", "capital-loss"]] + 1)
numerical = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
features[numerical] = MinMaxScaler().fit_transform(features[numerical])
features_final = pd.get_dummies(features)
X_train, X_test, y_train, y_test = train_test_split(
    features_final, income, test_size=0.2, random_state=0
)
X_test = X_test.iloc[:N_ROWS]
raw_test = data.loc[X_test.index]

# ---------- B) app pipeline, row by row ----------
def to_form(row):
    married = row["marital-status"].strip() == "Married-civ-spouse"
    return {
        "age": str(int(row["age"])),
        "education_num": str(int(row["education-num"])),
        "capital_gain": str(int(row["capital-gain"])),
        "capital_loss": str(int(row["capital-loss"])),
        "marital_status": "Married" if married else "Single",
    }

print(f"Comparing {len(X_test)} real test rows...\n")
all_ok = True
for name, model in P.MODELS.items():
    cols = list(model.feature_names_in_)
    notebook_pred = model.predict(X_test[cols])

    app_pred, worst_diff = [], 0.0
    for idx, (_, row) in enumerate(raw_test.iterrows()):
        df = P.preprocess(to_form(row))
        X_app = P.features_for(model, df)
        app_pred.append(int(model.predict(X_app)[0]))
        worst_diff = max(worst_diff, float(np.abs(X_app.values - X_test[cols].iloc[idx].values).max()))

    match = int((np.array(app_pred) == notebook_pred).sum())
    print(f"{name:28s} predictions match: {match}/{len(X_test)}   "
          f"largest feature difference: {worst_diff:.2e}")
    all_ok &= (match == len(X_test))

print("\nRESULT:", "PASS - the app gives the same predictions as the notebook" if all_ok
      else "MISMATCH - tell Claude the numbers above")