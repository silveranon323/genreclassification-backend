from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import io
import random  # Import random module for dummy genre selection

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
        print("No file part received")
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        print("No file selected")
        return jsonify({"error": "No selected file"}), 400

    # Read file in memory
    file_content = file.read()
    audio_buffer = io.BytesIO(file_content)

    print("\n--- Received Audio File ---")
    print(f"Filename: {file.filename}")
    print(f"File Size: {len(file_content)} bytes")
    print(f"File Type: {file.content_type}")
    print("---------------------------\n")

    try:
        # Load audio using librosa
        y, sr = librosa.load(audio_buffer, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        print("\n--- Audio Processing Details ---")
        print(f"Sample Rate: {sr}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Num Samples: {len(y)}")
        print("-------------------------------\n")

        # Select a random genre for demo purposes
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
        print(f"Error processing audio: {e}")
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500


if __name__ == "__main__":
    app.run()
