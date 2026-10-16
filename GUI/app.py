import logging
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, redirect, render_template, request, url_for

import preprocess as P

app = Flask(__name__)
log = logging.getLogger("charityml")

def form_context(**extra):
    context = {"marital_options": P.MARITAL_OPTIONS, "ranges": P.RANGES, "form": {}, "error": None}
    context.update(extra)
    return context

def summarize(results):
    votes = [r["prediction"] for r in results]
    if all(v == 1 for v in votes):
        return "Likely donor (income > $50K)", "donor"
    if all(v == 0 for v in votes):
        return "Unlikely donor (income <= $50K)", "not_donor"
    return "The models disagree about this person", "mixed"

@app.route("/")
def home():
    return render_template("index.html", **form_context())

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    try:
        results = P.predict_all(request.form)
    except ValueError as err:
        return render_template("index.html", **form_context(error=str(err), form=request.form)), 400
    except Exception:
        log.exception("Prediction failed")
        return render_template(
            "index.html",
            **form_context(error="Something went wrong on our side. Please try again.", form=request.form),
        ), 500

    ada_pred = ""
    gb_pred = ""
    ada_pred_numeric = 0
    gb_pred_numeric = 0

    for r in results:
        if "AdaBoost" in r["model"]:
            ada_pred_numeric = r["prediction"]
            ada_pred = ">50K" if ada_pred_numeric == 1 else "<=50K"
        elif "Gradient" in r["model"]:
            gb_pred_numeric = r["prediction"]
            gb_pred = ">50K" if gb_pred_numeric == 1 else "<=50K"

    try:
        user_x = float(request.form.get("age", 30))
    except (ValueError, TypeError):
        user_x = 30

    train_points = [
        {"x": 22, "y": 0}, {"x": 25, "y": 0}, {"x": 28, "y": 0},
        {"x": 35, "y": 0}, {"x": 42, "y": 0}, {"x": 38, "y": 1},
        {"x": 45, "y": 1}, {"x": 50, "y": 1}, {"x": 60, "y": 1}
    ]
    train_points_json = json.dumps(train_points)

    return render_template(
        "result.html", 
        ada_pred=ada_pred, 
        gb_pred=gb_pred,
        ada_pred_numeric=ada_pred_numeric,
        gb_pred_numeric=gb_pred_numeric,
        user_x=user_x,
        train_points=train_points_json
    )

@app.route("/health")
def health():
    return "ok"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)