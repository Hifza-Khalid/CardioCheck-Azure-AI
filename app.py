"""Flask app for the UCI Heart Disease (Cleveland) Perceptron model,
with Azure AI Foundry generating plain-language explanations."""
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
from flask import (Flask, jsonify, render_template, request,
                   send_from_directory, session)
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(HERE, "templates"),
            static_folder=os.path.join(HERE, "static"))
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
app.secret_key = "cardio-check-demo-secret"

# ---------- Azure AI Foundry ----------
AZURE_ENDPOINT = "https://hifza-foundry-resource.services.ai.azure.com/openai/v1"
AZURE_DEPLOYMENT = "gpt-5-mini"

# ⚠️ PASTE YOUR REAL API KEY HERE BETWEEN THE QUOTES:
AZURE_API_KEY = "api_key"

azure_client = OpenAI(
    base_url=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY
)
print("✅ Azure client initialized successfully!")

# ---------- Model ----------
with open(os.path.join(HERE, "model.json"), encoding="utf-8") as f:
    M = json.load(f)

FEATS = M["features"]
W = np.asarray(M["weights"], dtype=float)
B = float(M["bias"])
MU = np.asarray(M["mean"], dtype=float)
SD = np.asarray(M["std"], dtype=float)

BINARY_FEATURES = {"sex", "fbs", "exang"}
CATEGORICAL_NEUTRAL = {"cp": 0.0, "restecg": 0.0, "slope": 1.0, "ca": 0.0, "thal": 3.0}

LABELS = {
    "age": "Age", "sex": "Gender", "cp": "Chest pain type",
    "trestbps": "Resting BP", "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "exang": "Exercise angina",
    "oldpeak": "ST depression", "slope": "ST slope",
    "ca": "Major vessels", "thal": "Thal",
}

CATEGORY_LABELS = {
    "sex": {"0": "Female", "1": "Male"},
    "cp": {"1": "Typical angina", "2": "Atypical angina",
           "3": "Non-anginal pain", "4": "Asymptomatic"},
    "restecg": {"0": "Normal", "1": "ST-T abnormality", "2": "LV hypertrophy"},
    "slope": {"1": "Upsloping", "2": "Flat", "3": "Downsloping"},
    "thal": {"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"},
    "fbs": {"0": "No", "1": "Yes"},
    "exang": {"0": "No", "1": "Yes"},
}


def _to_numeric_value(feature, value):
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            return None
        token = raw.lower()
        if feature == "sex":
            if token in {"m", "male", "1"}:
                return 1.0
            if token in {"f", "female", "0"}:
                return 0.0
        if token in {"t", "true", "yes", "y", "1"}:
            return 1.0
        if token in {"f", "false", "no", "n", "0"}:
            return 0.0
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid value for '{feature}': {value!r}") from exc
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for '{feature}': {value!r}") from exc


def _neutral_raw_value(feature):
    if feature in BINARY_FEATURES:
        return 0.0
    if feature in CATEGORICAL_NEUTRAL:
        return CATEGORICAL_NEUTRAL[feature]
    return float(MU[FEATS.index(feature)])


def score(values):
    row = []
    for name in FEATS:
        x = _to_numeric_value(name, values.get(name))
        if x is None:
            x = _neutral_raw_value(name)
        if not math.isfinite(x):
            raise ValueError(f"{name} must be a finite number")
        row.append(x)
    z = (np.asarray(row, dtype=float) - MU) / SD
    return float(z @ W + B)


def sigmoid_confidence(s):
    x = s / 3.0
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def make_result(values):
    s = score(values)
    positive = s > 0
    return {
        "prediction": int(positive),
        "score": round(s, 6),
        "confidence": round(sigmoid_confidence(s), 6),
        "label": "Positive" if positive else "Negative",
    }


def pretty_values(values):
    rows = []
    for k in FEATS:
        v = values.get(k)
        if v is None or v == "":
            rows.append((LABELS[k], "—"))
            continue
        if k in CATEGORY_LABELS:
            rows.append((LABELS[k], CATEGORY_LABELS[k].get(str(v), str(v))))
        else:
            rows.append((LABELS[k], str(v)))
    return rows


def generate_ai_explanation(result, rows):
    """Ask Azure to explain the prediction using the Responses API."""
    if azure_client is None:
        return None

    feature_text = "\n".join(f"- {label}: {val}" for label, val in rows)
    direction = ("suggests possible signs of heart disease"
                 if result["prediction"] == 1
                 else "does not show strong signs of heart disease")

    prompt = f"""A user completed a heart-risk questionnaire.
Model raw score: {result['score']:.3f}
Prediction: {result['label']} ({direction})

Their answers:
{feature_text}

Write a short, warm, plain-language explanation (3-4 sentences) for the user.
Be honest that this is a learning demo, not a diagnosis. Do not give medical advice.
End with a gentle recommendation to see a clinician if they feel unwell."""

    try:
        response = azure_client.responses.create(
            model=AZURE_DEPLOYMENT,
            instructions="You are a helpful, cautious health-explanation assistant. Keep answers short and never diagnose.",
            input=prompt,
        )
        
        # The SDK provides .output_text automatically for Responses API
        return response.output_text.strip()

    except Exception as exc:
        print(f"\n❌ FATAL AZURE ERROR: {type(exc).__name__} - {exc}\n")
        return f"AZURE ERROR: {type(exc).__name__} - {exc}"


@app.route("/")
def landing():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("index.html")


@app.route("/chat")
def chat():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_form():
    data = request.form.to_dict()
    try:
        result = make_result(data)
    except (TypeError, ValueError) as exc:
        return render_template("index.html", error=str(exc)), 400

    rows = pretty_values({k: data.get(k, "") for k in FEATS})
    ai_explanation = generate_ai_explanation(result, rows)

    return render_template(
        "index.html",
        result=result,
        rows=rows,
        when=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=M,
        ai_explanation=ai_explanation,
    )


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object of feature values."), 400
    try:
        return jsonify(make_result(data))
    except (TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400


@app.route("/api/model")
def model_info():
    return jsonify(train=M.get("train"), test=M.get("test"), features=FEATS)


# ---------- Image Routes ----------
@app.route("/logo")
def logo():
    return send_from_directory(os.path.join(HERE, "images"), "logo.png")

@app.route("/mascot")
def mascot():
    return send_from_directory(os.path.join(HERE, "images"), "cardio.png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)