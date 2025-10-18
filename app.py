from flask import Flask, render_template, request
from dotenv import load_dotenv
import joblib
import numpy as np
import json
import os

# =======================
# Load Environment Variables
# =======================
load_dotenv()  # reads variables from .env

# Create Flask App
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")

# =======================
# Load Model & Scaler
# =======================
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# =======================
# Load Model Info (name + accuracy)
# =======================
model_info = {}
if os.path.exists("model_info.json"):
    with open("model_info.json", "r") as f:
        model_info = json.load(f)

# =======================
# Routes
# =======================
@app.route("/")
def home():
    model_name = model_info.get("model_name", "Unknown Model")
    accuracy = model_info.get("accuracy", "N/A")
    return render_template("index.html", model_name=model_name, accuracy=accuracy)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        data = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["blood_pressure"]),
            float(request.form["skin_thickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]

        # Scale and predict
        features_scaled = scaler.transform([data])
        prediction = model.predict(features_scaled)[0]

        result = "⚠️ High risk of Diabetes" if prediction == 1 else "✅ Low risk of Diabetes"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

# =======================
# Run App
# =======================
if __name__ == "__main__":
    debug_mode = os.getenv("DEBUG", "True").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
