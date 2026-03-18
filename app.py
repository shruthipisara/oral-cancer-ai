from flask import Flask, render_template, request, redirect, session
from ultralytics import YOLO
import os
import uuid
import gc
import requests

# ==============================
# App Configuration
# ==============================

app = Flask(__name__, static_folder="static")
app.secret_key = "securekey123"

# Required for Render (prevents Ultralytics permission issues)
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

# Folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==============================
# Hugging Face Model Download
# ==============================

MODEL_URL = "https://huggingface.co/shruthipisara/oral-cancer-ai/resolve/main/oral_cancer_model.pt"
MODEL_PATH = "oral_cancer_model.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully.")

# Lazy loading (important for memory)
model = None

# ==============================
# Routes
# ==============================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "doctor" and password == "admin123":
            session["user"] = username
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

    if "image" not in request.files:
        return redirect("/dashboard")

    file = request.files["image"]

    if file.filename == "":
        return redirect("/dashboard")

    try:
        # Load model only when needed
        if model is None:
            model = YOLO(MODEL_PATH)

        # Save uploaded image
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Run prediction (CPU for Render free plan)
        results = model.predict(
            source=filepath,
            conf=0.35,
            imgsz=640,
            device="cpu"
        )

        # Default values
        risk = "Healthy / No Cancer Detected"
        cancer_status = "NO"
        confidence_percent = 0

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                highest_conf = max(float(box.conf) for box in r.boxes)

                confidence_percent = round(highest_conf * 100, 2)
                cancer_status = "YES"

                if confidence_percent >= 75:
                    risk = "High Risk Lesion Detected"
                elif confidence_percent >= 50:
                    risk = "Moderate Risk - Clinical Review Recommended"
                else:
                    risk = "Low Suspicion - Monitor Patient"

        # Save result image with bounding boxes
        result_path = os.path.join(RESULT_FOLDER, filename)
        results[0].save(filename=result_path)

        # Free memory
        gc.collect()

        return render_template(
            "result.html",
            image=filename,
            confidence=confidence_percent,
            risk=risk,
            cancer_status=cancer_status
        )

    except Exception as e:
        print("ERROR:", str(e))
        return "Internal Server Error"


# ==============================
# Render Entry Point
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)