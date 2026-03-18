from flask import Flask, render_template, request, redirect, session
from ultralytics import YOLO
import os
import uuid
import gc
import requests

app = Flask(__name__)
app.secret_key = "securekey123"

# Render fix
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

MODEL_URL = "https://huggingface.co/shruthipisara/oral-cancer-ai/resolve/main/oral_cancer_model.pt"
MODEL_PATH = "oral_cancer_model.pt"

model = None


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "doctor" and request.form["password"] == "admin123":
            session["user"] = "doctor"
            return redirect("/dashboard")
        else:
            return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model

    if "user" not in session:
        return redirect("/")

    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect("/dashboard")

    try:
        # ✅ DOWNLOAD MODEL HERE ONLY (SAFE)
        if not os.path.exists(MODEL_PATH):
            print("Downloading model...")
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

        # ✅ LOAD MODEL HERE ONLY
        if model is None:
            model = YOLO(MODEL_PATH)

        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        results = model.predict(source=filepath, conf=0.35, imgsz=640, device="cpu")

        risk = "Healthy / No Cancer Detected"
        cancer_status = "NO"
        confidence_percent = 0

        for r in results:
            if r.boxes and len(r.boxes) > 0:
                conf = max(float(b.conf) for b in r.boxes)
                confidence_percent = round(conf * 100, 2)
                cancer_status = "YES"

                if confidence_percent >= 75:
                    risk = "High Risk Lesion Detected"
                elif confidence_percent >= 50:
                    risk = "Moderate Risk - Clinical Review Recommended"
                else:
                    risk = "Low Suspicion - Monitor Patient"

        result_path = os.path.join(RESULT_FOLDER, filename)
        results[0].save(filename=result_path)

        gc.collect()

        return render_template(
            "result.html",
            image=filename,
            confidence=confidence_percent,
            risk=risk,
            cancer_status=cancer_status
        )

    except Exception as e:
        print("ERROR:", e)
        return "Error occurred"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)