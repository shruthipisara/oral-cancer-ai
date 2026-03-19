from flask import Flask, render_template, request, redirect, session
import os
import uuid
import requests

app = Flask(__name__)
app.secret_key = "securekey123"

# =========================
# FOLDERS
# =========================
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =========================
# HUGGING FACE API URL (CORRECT)
# =========================
HF_API_URL = "https://shruthipisara-oral-cancer-app.hf.space/run/predict"

# =========================
# LOGIN (DEMO MODE - ANY USER)
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form["username"]
        return redirect("/dashboard")
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
# PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect("/dashboard")

    try:
        # -------------------------
        # Save uploaded image
        # -------------------------
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # -------------------------
        # Send to Hugging Face
        # -------------------------
        with open(filepath, "rb") as img:
            response = requests.post(
                HF_API_URL,
                files={"data": img}
            )

        result = response.json()
        print("HF RESPONSE:", result)

        # -------------------------
        # Extract OUTPUTS
        # -------------------------
        output_image = result.get("data", ["", ""])[0]   # annotated image (base64)
        output_text = result.get("data", ["", ""])[1]    # prediction text

        # -------------------------
        # SAVE RESULT IMAGE (from HF)
        # -------------------------
        import base64

        if output_image.startswith("data:image"):
            img_data = output_image.split(",")[1]
            img_bytes = base64.b64decode(img_data)

            result_filename = "result_" + filename
            result_path = os.path.join(RESULT_FOLDER, result_filename)

            with open(result_path, "wb") as f:
                f.write(img_bytes)
        else:
            # fallback if image not returned
            result_filename = filename

        # -------------------------
        # DECISION LOGIC
        # -------------------------
        if "Cancer" in output_text:
            cancer_status = "YES"
            risk = "High Risk Lesion Detected"
        else:
            cancer_status = "NO"
            risk = "Healthy / No Cancer Detected"

        # -------------------------
        # RETURN RESULT PAGE
        # -------------------------
        return render_template(
            "result.html",
            image=result_filename,
            confidence=output_text,
            risk=risk,
            cancer_status=cancer_status
        )

    except Exception as e:
        print("ERROR:", str(e))
        return "Server Error - Check logs"


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
