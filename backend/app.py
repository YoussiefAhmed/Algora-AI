from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime
import json
import os

app = Flask(__name__)
CORS(app)

# ===== Load AI Model =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "../model/algae_model.pkl"))


# ===== Serve Frontend =====
@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/<path:path>")
def files(path):
    return send_from_directory("../frontend", path)


# ===== Logging =====
def save_log(data):
    log = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **data
    }

    if not os.path.exists("logs.json"):
        with open("logs.json", "w") as f:
            json.dump([], f)

    with open("logs.json", "r") as f:
        old = json.load(f)

    old.append(log)

    with open("logs.json", "w") as f:
        json.dump(old, f, indent=3)


# ===== Input Validation =====
def sanity_check(d):
    try:
        float(d["temp"])
        float(d["light"])
        float(d["ph"])
    except:
        return "Values must be numeric only"
    return None


# ======================================================
# ========== المنطق الجديد الذكي =======================
# ======================================================

def ai_based_status(pred, prob, t, l, p, algae_type):

    if algae_type == "spirulina":
        perfect = (27 <= t <= 32 and 350 <= l <= 800 and 8.2 <= p <= 9.8)
        near    = (25 <= t <= 34 and 300 <= l <= 900 and 7.8 <= p <= 10.2)
    else:
        perfect = (24 <= t <= 30 and 250 <= l <= 700 and 6.8 <= p <= 7.8)
        near    = (22 <= t <= 32 and 200 <= l <= 800 and 6.5 <= p <= 8.2)

    # 1) لو القيم مثالية
    if perfect:
        if prob >= 50:
            return "healthy"
        else:
            return "healthy (low confidence)"

    # 2) لو قريبة من المثالي
    if near:
        if prob >= 50:
            return "healthy (minor deviations)"
        elif prob >= 35:
            return "monitor closely"
        else:
            return "not healthy"

    # 3) خارج الرينجات
    return "not healthy"


# ======================================================
# ===== Explainable AI  ================================
# ======================================================

def explain_ai(t, l, p, algae_type, prob):

    reasons = []
    reasons.append(f"AI confidence level: {prob}%")

    if algae_type == "spirulina":
        reasons.append(
            "Temp good for Spirulina" if 27<=t<=32 else "Temp not ideal for Spirulina"
        )
        reasons.append(
            "Light suitable" if 350<=l<=800 else "Light may affect growth"
        )
        reasons.append(
            "pH compatible" if 8.2<=p<=9.8 else "pH stressful for cells"
        )

    else:
        reasons.append(
            "Temp good for Chlorella" if 24<=t<=30 else "Temp not ideal for Chlorella"
        )
        reasons.append(
            "Light suitable" if 250<=l<=700 else "Light may affect growth"
        )
        reasons.append(
            "pH compatible" if 6.8<=p<=7.8 else "pH stressful for cells"
        )

    return reasons


# ======================================================
# ===== Recommendations ================================
# ======================================================

def recommend_ai(t, l, p, algae_type):

    rec = []

    if algae_type == "spirulina":

        if t < 27: rec.append("Increase temp to 27–32°C")
        if t > 32: rec.append("Reduce temp to 27–32°C")

        if l < 350: rec.append("Increase light to 350–800 Lux")
        if l > 800: rec.append("Reduce light intensity")

        if p < 8.2: rec.append("Raise pH slightly")
        if p > 9.8: rec.append("Lower pH slightly")

    else:

        if t < 24: rec.append("Increase temp to 24–30°C")
        if t > 30: rec.append("Reduce temp to 24–30°C")

        if l < 250: rec.append("Increase light to 250–700 Lux")
        if l > 700: rec.append("Reduce light intensity")

        if p < 6.8: rec.append("Raise pH slightly")
        if p > 7.8: rec.append("Lower pH slightly")

    if not rec:
        rec.append("Maintain current conditions")

    return rec


# ======================================================
# ================== API ===============================
# ======================================================

@app.route("/predict", methods=["POST"])
def predict():

    d = request.json

    err = sanity_check(d)
    if err:
        return jsonify({
            "status": "not healthy",
            "confidence": 100,
            "explain": ["Invalid input"],
            "recommend": [err]
        })

    t = float(d["temp"])
    l = float(d["light"])
    p = float(d["ph"])
    algae_type = d.get("type", "spirulina")

    x = pd.DataFrame([{
        "temp": t,
        "light": l,
        "ph": p
    }])

    pred = model.predict(x)[0]
    prob = round(float(model.predict_proba(x)[0].max())*100, 2)

    status = ai_based_status(pred, prob, t, l, p, algae_type)

    conf = prob if status != "not healthy" else None

    result = {
        "status": status,
        "confidence": conf,
        "type": algae_type,
        "explain": explain_ai(t, l, p, algae_type, prob),
        "recommend": recommend_ai(t, l, p, algae_type)
    }

    save_log(result)

    return jsonify(result)


@app.route("/logs")
def logs():
    if not os.path.exists("logs.json"):
        return jsonify([])
    with open("logs.json") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    app.run(debug=True)