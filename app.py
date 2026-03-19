from flask import Flask, render_template, request, redirect, session
import os
import uuid
import requests

app = Flask(__name__)
app.secret_key = "securekey123"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 👉 IMPORTANT: Replace this with your Hugging Face Space URL
HF_API_URL = "https://YOUR-SPACE-URL/predict"

# =========================
# ROUTES
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


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect("/dashboard")

    try:
        # Save uploaded image locally (for preview if needed)
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 🔥 Send image to Hugging Face Space
        with open(filepath, "rb") as img:
            response = requests.post(
                HF_API_URL,
                files={"file": img}
            )

        result_text = response.text.strip()

        # Simple mapping
        if "Cancer" in result_text:
            cancer_status = "YES"
            risk = "High Risk Lesion Detected"
        else:
            cancer_status = "NO"
            risk = "Healthy / No Cancer Detected"

        return render_template(
            "result.html",
            image=filename,   # still show uploaded image
            confidence="N/A",
            risk=risk,
            cancer_status=cancer_status
        )

    except Exception as e:
        print("ERROR:", str(e))
        return "Server Error - Check logs"


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
