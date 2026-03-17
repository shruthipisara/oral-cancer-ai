from flask import Flask, render_template, request, redirect, session
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
app.secret_key = "securekey123"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load trained YOLO model
model = YOLO("oral_cancer_model.pt")


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

    if "user" not in session:
        return redirect("/")

    if "image" not in request.files:
        return redirect("/dashboard")

    file = request.files["image"]

    if file.filename == "":
        return redirect("/dashboard")

    # Generate unique filename
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # YOLO prediction
    results = model(
        filepath,
        conf=0.35,   # Slightly lower threshold for early detection
        imgsz=640,
        device="cpu",
        half=False
    )

    # Default assumption
    risk = "Healthy / No Cancer Detected"
    cancer_status = "NO"
    confidence_percent = 0

    # Process detection results
    for r in results:
        if len(r.boxes) > 0:

            # Get highest confidence detection
            highest_conf = max(float(box.conf) for box in r.boxes)

            confidence_percent = round(highest_conf * 100, 2)
            cancer_status = "YES"

            # Risk classification
            if confidence_percent >= 75:
                risk = "High Risk Lesion Detected"
            elif confidence_percent >= 50:
                risk = "Moderate Risk - Clinical Review Recommended"
            else:
                risk = "Low Suspicion - Monitor Patient"

    # Save image with bounding boxes
    result_path = os.path.join(RESULT_FOLDER, filename)
    results[0].save(filename=result_path)

    return render_template(
        "result.html",
        image=filename,
        confidence=confidence_percent,
        risk=risk,
        cancer_status=cancer_status
    )


# Run server (important for Render deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)