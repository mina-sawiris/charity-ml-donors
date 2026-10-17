# CharityML – Donor Potential Web App

A Flask web app that predicts whether a person earns **more than $50,000 a year**,
so a charity can prioritise outreach to likely donors. It serves two models trained in
the *Finding Donors for CharityML* project (AdaBoost and Gradient Boosting) and shows
both predictions side by side.

**Live demo:** _add your Render link here_

## How it works

1. The user fills in a form: age, education level, marital status, capital gain, capital loss.
2. `preprocess.py` validates the input, log-transforms capital gain/loss and scales the
   numeric columns with the saved `scaler.pkl`.
3. Each saved model returns a prediction (`>50K` or `<=50K`).
4. The result page shows both predictions.

## Project structure

```
charity-ml-donors/
├── GUI/
│   ├── app.py                # Flask routes: /, /predict, /health
│   └── templates/            # index.html (form), result.html (prediction)
├── preprocess.py             # Validation, preprocessing, prediction
├── schema.json               # Form fields, ranges, scaler limits, model features
├── scaler.pkl                # Fitted scaler
├── reduced_adaboost_model_mina.pkl
├── reduced_GBmodel_heba.pkl
├── finding_donors-*.ipynb    # Training notebooks
├── census.csv                # Dataset
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/mina-sawiris/charity-ml-donors.git
cd charity-ml-donors

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python GUI/app.py
```

Open http://127.0.0.1:5000 in your browser.

## Deployment (Render)

| Setting | Value |
|---------|-------|
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn --chdir GUI app:app` |
| Health check path | `/health` |

## Tech stack

Python, Flask, scikit-learn, pandas, NumPy, Gunicorn, Render

## Team

| Role | Members |
|------|---------|
| Frontend | Mina, Heba |
| Backend | Shahd, Nemat |
| DevOps | Youssef |
