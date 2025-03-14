from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import io
import random

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins

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


@app.route("/api/predict", methods=["POST"])
def predict():
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

        return jsonify(
            {
                "message": "Audio file received successfully",
                "filename": file.filename,
                "duration": duration,
                "predicted_genre": predicted_genre,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500


if __name__ == "__main__":
    app.run()
