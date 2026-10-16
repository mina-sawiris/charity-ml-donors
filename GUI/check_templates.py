"""
check_templates.py - run this after Mina and Heba's files are in GUI/templates and GUI/static.

    python GUI/check_templates.py          (from the repo root)

It reads their HTML and also submits the form through the real Flask app, then prints
PASS / FAIL for each check with a hint about what to ask the Frontend team to fix.
"""
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import preprocess as P  # noqa: E402
from app import app  # noqa: E402

TEMPLATES = os.path.join(HERE, "templates")
STATIC = os.path.join(HERE, "static")
VALID = {"age": "39", "education_num": "13", "capital_gain": "2174",
         "capital_loss": "0", "marital_status": "Single"}

passed = failed = 0


def check(ok, name, hint=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}" + (f"\n        -> {hint}" if hint else ""))
    return ok


def read(name):
    path = os.path.join(TEMPLATES, name)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


# ------------------------------------------------------------------ files exist
index, result = read("index.html"), read("result.html")
have_files = check(bool(index.strip()), "templates/index.html exists and is not empty",
                   "the file is missing or empty")
have_files &= check(bool(result.strip()), "templates/result.html exists and is not empty",
                    "the file is missing or empty")
if not have_files:
    print("\nStopped: the templates are not there yet.")
    raise SystemExit(1)

# ------------------------------------------------------------------ index.html
print("\n-- index.html (the form) --")
form_tag = re.search(r"<form[^>]*>", index, re.I | re.S)
tag = form_tag.group(0) if form_tag else ""
check(bool(form_tag) and re.search(r"method\s*=\s*[\"']post[\"']", tag, re.I),
      'form uses method="post"', 'write <form method="post" action="/predict">')
check("/predict" in tag or "url_for('predict')" in tag or 'url_for("predict")' in tag,
      "form action points to /predict", 'action="/predict" (or action="{{ url_for(\'predict\') }}")')

names = set(re.findall(r"""(?<![\w-])name\s*=\s*["']([^"']+)["']""", index))
for field in P.FORM_FIELDS:
    check(field in names, f'input name="{field}" exists',
          f"the field must be called exactly '{field}' (found: {sorted(names)})")
extra = names - set(P.FORM_FIELDS)
check(not extra, "no unexpected field names", f"these names are not used by the backend: {sorted(extra)}")

for option in P.MARITAL_OPTIONS:
    check(option in index or "marital_options" in index,
          f"marital status offers '{option}'",
          f"the dropdown/radio values must be exactly {P.MARITAL_OPTIONS} (or loop over marital_options)")
check(re.search(r"\{\{\s*error", index) is not None, "index.html shows {{ error }}",
      "add {% if error %}<div class='alert'>{{ error }}</div>{% endif %} so users see why it failed")
check("form" in re.findall(r"\bform\b\.get|\bform\[", index) or "form.get" in index or "form[" in index,
      "index.html keeps what the user typed (form.get(...))",
      'add value="{{ form.get(\'age\', \'\') }}" to the inputs so a mistake does not erase everything')

def css_name(href):
    """File name of a stylesheet link, for plain paths and for url_for('static', filename=...)."""
    m = re.search(r"""filename\s*=\s*['"]([^'"]+)['"]""", href)
    return m.group(1) if m else os.path.basename(href)


css_files = sorted({css_name(h) for h in re.findall(r"""href\s*=\s*["']([^"']*\.css[^"']*)["']""", index + result)
                    if ".css" in h} | {css_name(h) for h in re.findall(r"""href\s*=\s*"([^"]*\.css[^"]*)\"""", index + result)})
check(bool(css_files), "pages link a CSS file",
      "add <link rel=\"stylesheet\" href=\"{{ url_for('static', filename='styles.css') }}\">")
for fname in css_files:
    check(os.path.exists(os.path.join(STATIC, fname)), f"CSS file static/{fname} exists",
          "the name in the HTML must match the file in GUI/static (styles.css vs style.css?)")

# ------------------------------------------------------------------ result.html
print("\n-- result.html (the answer) --")
check(re.search(r"\{%\s*for\s+\w+\s+in\s+results", result) is not None,
      "result.html loops over results (one card per model)",
      "use {% for r in results %} ... {{ r.model }} {{ r.label }} {{ r.probability }}% {% endfor %}")
check("probability" in result, "result.html shows the probability", "show {{ r.probability }}")
check("model" in result, "result.html shows the model name", "show {{ r.model }}")
check("{{ label" in result or "label" in result, "result.html shows the summary or labels",
      "show {{ label }} or {{ r.label }}")
check("href=\"/\"" in result or "url_for('home')" in result or 'url_for("home")' in result or "href='/'" in result,
      "result.html has a link back to the form", "add <a href=\"/\">Try another person</a>")

# ------------------------------------------------------------------ end-to-end
print("\n-- end-to-end through the real app --")
client = app.test_client()
r = client.get("/")
check(r.status_code == 200, "GET / returns 200", f"status {r.status_code}")
r = client.post("/predict", data=VALID)
body = r.get_data(as_text=True)
check(r.status_code == 200, "valid form returns 200", f"status {r.status_code}")
for model_name in P.MODELS:
    check(model_name in body, f"result page shows the '{model_name}' result",
          "result.html must print every model's name with {{ r.model }}")
bad = client.post("/predict", data=dict(VALID, age="500"))
bad_body = bad.get_data(as_text=True)
check(bad.status_code == 400, "age=500 returns 400", f"status {bad.status_code}")
check("between" in bad_body, "the error message is visible on the page",
      "index.html must print {{ error }}")
check("500" in bad_body, "the wrong value stays in the form", "inputs need value=\"{{ form.get('age', '') }}\"")
empty = client.post("/predict", data={})
check(empty.status_code == 400 and "required" in empty.get_data(as_text=True),
      "empty form shows the 'required' messages")
check(client.post("/predict", data=dict(VALID, marital_status="Divorced")).status_code == 400,
      "a value outside the allowed options is rejected")
for fname in css_files:
    check(client.get(f"/static/{fname}").status_code == 200, f"/static/{fname} is served",
          "the CSS file must be inside GUI/static/")

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(0 if failed == 0 else 1)
