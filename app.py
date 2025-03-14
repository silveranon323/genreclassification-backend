from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import librosa
import numpy as np
import io
import random

app = Flask(__name__)

# Allow all origins, methods, and headers
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

GENRES = [
    "Pop",
    "Rock",
    "Hip-Hop",
    "Jazz",
    "Classical",
    "Electronic",
    "Reggae",
    "Blues",
    "Metal",
    "Country",
    "Folk",
    "R&B",
    "Funk",
    "Disco",
    "Punk",
]

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":  # Handle preflight request
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        audio_buffer = io.BytesIO(file.read())
        y, sr = librosa.load(audio_buffer, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        predicted_genre = random.choice(GENRES)

        response = jsonify(
            {
                "message": "Audio file received successfully",
                "filename": file.filename,
                "duration": duration,
                "predicted_genre": predicted_genre,
            }
        )
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        response = jsonify({"error": f"Failed to process audio: {str(e)}"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
