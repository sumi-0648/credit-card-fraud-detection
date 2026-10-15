
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

bundle = joblib.load(BASE_DIR / "fraud_model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD = bundle["threshold"]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    values = {feature: "0" for feature in FEATURES}

    if request.method == "POST":
        try:
            row = {}

            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()

                if not raw:
                    raise ValueError(f"Enter a value for {feature}.")

                value = float(raw)

                if not pd.notna(value):
                    raise ValueError(f"Invalid value for {feature}.")

                if feature in ("Time", "Amount") and value < 0:
                    raise ValueError(f"{feature} cannot be negative.")

                row[feature] = value
                values[feature] = raw

            transaction = pd.DataFrame([row], columns=FEATURES)
            probability = float(model.predict_proba(transaction)[0, 1])

            is_fraud = probability >= THRESHOLD

            result = {
                "is_fraud": is_fraud,
                "label": (
                    "Potential Fraud Detected"
                    if is_fraud
                    else "Likely Legitimate"
                ),
                "probability": round(probability * 100, 2),
            }

        except (ValueError, TypeError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        features=FEATURES,
        values=values,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)

