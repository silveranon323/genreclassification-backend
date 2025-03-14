from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import io
import random
import os  # Import os for environment variables

app = Flask(__name__)
CORS(app)

# List of sample genres from Spotify
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

    # Read file in memory
    file_content = file.read()
    audio_buffer = io.BytesIO(file_content)

    try:
        # Load audio using librosa
        y, sr = librosa.load(audio_buffer, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Select a random genre for demo purposes
        predicted_genre = random.choice(GENRES)

        return jsonify(
            {
                "message": "Audio file received successfully",
                "filename": file.filename,
                "duration": round(duration, 2),
                "predicted_genre": predicted_genre,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render-assigned PORT
    app.run(host="0.0.0.0", port=port)
