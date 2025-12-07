import os
import numpy as np
import joblib
import requests
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, make_response
from groq import Groq
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # disable static cache


# ======================================================
# LOAD MODELS
# ======================================================
print("\n🔥 Starting Smart AgroGuide... Loading models...\n")

try:
    crop_model = joblib.load("models/crop_model.pkl")
    crop_label_encoder = joblib.load("models/crop_label_encoder.pkl")
    print("✔ Crop model loaded.")
except Exception as e:
    print("❌ Crop Model Load Error:", e)
    crop_model, crop_label_encoder = None, None

try:
    disease_model = load_model("models/disease_model.h5")
    print("✔ Disease model loaded.")
except Exception as e:
    print("❌ Disease Model Load Error:", e)
    disease_model = None

try:
    class_names = joblib.load("models/disease_label_encoder.pkl")
    print("✔ Disease labels loaded.")
except Exception as e:
    print("❌ Disease Label Load Error:", e)
    class_names = []


# ======================================================
# GROQ CLIENT
# ======================================================
GROQ_API_KEY = "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"

def get_groq_client():
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print("Groq init error:", e)
        return None



# ======================================================
# DISABLE CACHE (IMPORTANT)
# ======================================================
def nocache(template_name):
    response = make_response(render_template(template_name))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response



# ======================================================
# WARM-UP FIX (solves popup & first-time error)
# ======================================================
_warmed_up = False

@app.before_request
def warmup():
    global _warmed_up
    if not _warmed_up:
        print("\n⚡ Running Groq warm-up...\n")
        try:
            client = get_groq_client()
            if client:
                client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Hello"}]
                )
                print("✔ Warm-up complete. Chatbot ready!\n")
        except Exception as e:
            print("Warm-up failed:", e)
        _warmed_up = True



# ======================================================
# ROUTES
# ======================================================
@app.route("/")
def home():
    return nocache("index.html")

@app.route("/chatbot")
def chatbot_page():
    return nocache("chatbot.html")

@app.route("/weather")
def weather_page():
    return nocache("weather.html")

@app.route("/crop-recommend")
def crop_recommend_page():
    return nocache("crop.html")

@app.route("/disease")
def disease_page():
    return nocache("disease.html")

@app.route("/about")
def about_page():
    return nocache("about.html")



# ======================================================
# TEXT CHATBOT API
# ======================================================
@app.route("/api/chat", methods=["POST"])
def ask_bot():
    message = request.form.get("message", "")

    client = get_groq_client()
    if client is None:
        return jsonify({"answer": "Chatbot cannot initialize"}), 500

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an intelligent agriculture assistant."},
                {"role": "user", "content": message}
            ]
        )
        reply = completion.choices[0].message.content
        return jsonify({"answer": reply})

    except Exception as e:
        print("Chatbot ERROR:", e)
        return jsonify({"answer": "Server communication failed"}), 500



# ======================================================
# AUDIO CHATBOT (Whisper → LLaMA)
# ======================================================
@app.route("/api/audio-chat", methods=["POST"])
def audio_chat():
    if "audio" not in request.files:
        return jsonify({"answer": "No audio uploaded"}), 400

    audio = request.files["audio"]
    filename = secure_filename(audio.filename)
    path = f"temp_audio_{filename}"
    audio.save(path)

    client = get_groq_client()
    if client is None:
        return jsonify({"answer": "Chatbot offline"}), 500

    try:
        # 1️⃣ Transcribe
        with open(path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                response_format="json"
            )
        text = transcription.text
        print("🎤 Transcript:", text)

        # 2️⃣ Generate response
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an intelligent agriculture assistant."},
                {"role": "user", "content": text}
            ]
        )
        reply = completion.choices[0].message.content

        os.remove(path)

        return jsonify({"answer": reply, "transcript": text})

    except Exception as e:
        print("AudioChat ERROR:", e)
        return jsonify({"answer": "Audio processing failed"}), 500



# ======================================================
# WEATHER API
# ======================================================
@app.route("/api/weather", methods=["POST"])
def get_weather():
    data = request.get_json()
    city = data.get("city")

    API_KEY = "86279deb774c5173ec41346bd870a71f"
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"

    try:
        r = requests.get(url)
        return jsonify(r.json())
    except Exception as e:
        print("Weather ERROR:", e)
        return jsonify({"error": "Weather API failed"}), 500



# ======================================================
# CROP RECOMMENDATION
# ======================================================
@app.route("/api/crop-recommend", methods=["POST"])
def recommend_crop():
    try:
        data = request.get_json()

        values = [
            float(data["N"]), float(data["P"]), float(data["K"]),
            float(data["temperature"]), float(data["humidity"]),
            float(data["ph"]), float(data["rainfall"])
        ]

        pred = crop_model.predict([values])[0]
        crop = crop_label_encoder.inverse_transform([pred])[0]

        return jsonify({"crop": crop})

    except Exception as e:
        print("Crop ERROR:", e)
        return jsonify({"error": "Invalid input"}), 400



# ======================================================
# DISEASE DETECTION
# ======================================================
@app.route("/api/disease-detect", methods=["POST"])
def disease_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_file.save("temp.jpg")

    try:
        img = image.load_img("temp.jpg", target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = disease_model.predict(x)
        idx = int(np.argmax(preds[0]))
        label = class_names[idx]
        confidence = float(np.max(preds[0])) * 100

        os.remove("temp.jpg")

        return jsonify({
            "result": label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("Disease ERROR:", e)
        return jsonify({"error": "Prediction failed"}), 500



# ======================================================
# RUN SERVER
# ======================================================
if __name__ == "__main__":
    print("\n🚀 Smart AgroGuide Running at http://127.0.0.1:5000\n")
    app.run(debug=False)
