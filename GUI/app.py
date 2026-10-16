import logging
import os
import sys

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

    label, outcome = summarize(results)
    return render_template("result.html", results=results, label=label, outcome=outcome, inputs=request.form)

@app.route("/health")
def health():
    return "ok"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)