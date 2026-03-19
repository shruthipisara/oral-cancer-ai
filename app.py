from flask import Flask, render_template, request, redirect, session
import os
import uuid
import requests

app = Flask(__name__)
app.secret_key = "securekey123"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ YOUR REAL HF SPACE API
HF_API_URL = "https://shruthipisara-oral-cancer-app.hf.space/run/predict"

# =========================
# LOGIN
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "doctor" and request.form["password"] == "admin123":
            session["user"] = "doctor"
            return redirect("/dashboard")
        else:
            return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html")

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect("/dashboard")

    try:
        # Save image
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 🔥 Send to Hugging Face
        with open(filepath, "rb") as img:
            response = requests.post(
                HF_API_URL,
                json={
                    "data": [img.read()]
                }
            )

        result = response.json()

        # 👉 Extract prediction safely
        output = result["data"][0]

        # Simple logic (based on your model output)
        if "Cancer" in str(output):
            cancer_status = "YES"
            risk = "High Risk Lesion Detected"
        else:
            cancer_status = "NO"
            risk = "Healthy / No Cancer Detected"

        return render_template(
            "result.html",
            image=filename,
            confidence="From AI Model",
            risk=risk,
            cancer_status=cancer_status
        )

    except Exception as e:
        print("ERROR:", e)
        return "Server Error"

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
