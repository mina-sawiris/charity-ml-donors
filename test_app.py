import os
import sys
import unittest

from jinja2 import DictLoader

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "GUI"))

import preprocess as P  
from app import app  

VALID = {"age": "39", "education_num": "13", "capital_gain": "2174",
         "capital_loss": "0", "marital_status": "Single"}
HIGH = {"age": "45", "education_num": "16", "capital_gain": "15000",
        "capital_loss": "0", "marital_status": "Married"}
LOW = {"age": "20", "education_num": "7", "capital_gain": "0",
       "capital_loss": "0", "marital_status": "Single"}

# simple stand-in pages if the real templates are not there yet
if not os.path.exists(os.path.join(BASE, "GUI", "templates", "index.html")):
    app.jinja_loader = DictLoader({
        "index.html": "FORM {{ error or '' }}",
        "result.html": "RESULT {{ label }}",
    })


def form(**changes):
    f = dict(VALID)
    f.update(changes)
    return f


class PreprocessTests(unittest.TestCase):
    def test_01_output_shape(self):
        df = P.preprocess(form())
        self.assertEqual(df.shape, (1, 5))
        for model in P.MODELS.values():
            self.assertEqual(P.features_for(model, df).shape, (1, 5))

    def test_02_column_order_matches_each_model(self):
        df = P.preprocess(form())
        for model in P.MODELS.values():
            self.assertEqual(list(P.features_for(model, df).columns),
                             list(model.feature_names_in_))

    def test_03_zero_capital_stays_zero(self):
        df = P.preprocess(form(capital_gain="0", capital_loss="0"))
        self.assertEqual(df["capital-gain"][0], 0.0)
        self.assertEqual(df["capital-loss"][0], 0.0)

    def test_04_scaling_range_and_boundaries(self):
        lo = P.preprocess(form(age="17", education_num="1"))
        hi = P.preprocess(form(age="100", education_num="16"))
        self.assertAlmostEqual(lo["age"][0], 0.0)
        self.assertAlmostEqual(lo["education-num"][0], 0.0)
        self.assertEqual(hi["age"][0], 1.0)              # clipped (max age in data is 90)
        self.assertAlmostEqual(hi["education-num"][0], 1.0)
        for col in P.SCALED_COLS:
            self.assertTrue(0.0 <= P.preprocess(form())[col][0] <= 1.0)

    def test_05_marital_encoding(self):
        self.assertEqual(P.preprocess(form(marital_status="Married"))[P.MARITAL_COL][0], 1.0)
        self.assertEqual(P.preprocess(form(marital_status="Single"))[P.MARITAL_COL][0], 0.0)

    def test_06_education_changes_vector(self):
        a = P.preprocess(form(education_num="1"))
        b = P.preprocess(form(education_num="16"))
        self.assertFalse(a.equals(b))

    def test_07_invalid_age_rejected(self):
        for bad in ["5", "200", "abc", "-3", "30.5"]:
            with self.assertRaises(ValueError, msg=bad):
                P.preprocess(form(age=bad))

    def test_08_empty_field_rejected(self):
        for field in P.FORM_FIELDS:
            with self.assertRaises(ValueError, msg=field):
                P.preprocess(form(**{field: ""}))

    def test_09_unknown_marital_value_rejected(self):
        with self.assertRaises(ValueError):
            P.preprocess(form(marital_status="Divorced"))

    def test_10_models_give_sensible_predictions(self):
        high = {r["model"]: r for r in P.predict_all(HIGH)}
        low = {r["model"]: r for r in P.predict_all(LOW)}
        for name in P.MODELS:
            self.assertEqual(high[name]["prediction"], 1, name)
            self.assertEqual(low[name]["prediction"], 0, name)
            self.assertTrue(0 <= high[name]["probability"] <= 100)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_11_home_page(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_12_valid_prediction(self):
        self.assertEqual(self.client.post("/predict", data=form()).status_code, 200)

    def test_13_invalid_input_returns_400(self):
        self.assertEqual(self.client.post("/predict", data={}).status_code, 400)
        self.assertEqual(self.client.post("/predict", data=form(age="500")).status_code, 400)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\n{passed}/{result.testsRun} tests passed")