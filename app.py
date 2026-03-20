import os
import sys
import json
import glob
from flask import Flask, render_template, request, jsonify, send_file

sys.path.insert(0, ".")
from src.generate import generate_song, midi_to_mp3

app = Flask(__name__)

GENERATED_DIR = "outputs/generated"
SF2_PATH      = "assets/FluidR3.sf3"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Receives settings from the UI, generates a song, converts to MP3.
    Returns the filename and metadata as JSON.
    """
    data         = request.get_json()
    num_notes    = int(data.get("num_notes",   200))
    temperature  = float(data.get("temperature", 0.8))
    note_duration= float(data.get("note_duration", 0.5))

    # Build a unique filename based on settings
    existing = glob.glob(os.path.join(GENERATED_DIR, "song_*.mp3"))
    next_num = len(existing) + 1
    filename = f"song_{next_num:02d}"

    midi_path = os.path.join(GENERATED_DIR, f"{filename}.mid")
    mp3_path  = os.path.join(GENERATED_DIR, f"{filename}.mp3")

    try:
        # Generate MIDI
        generate_song(
            output_path    = midi_path,
            num_notes      = num_notes,
            temperature    = temperature,
            sequence_length= 64
        )

        # Convert to MP3
        midi_to_mp3(
            midi_path  = midi_path,
            sf2_path   = SF2_PATH,
            output_mp3 = mp3_path
        )

        # Get duration via file size estimate
        size_kb = os.path.getsize(mp3_path) / 1024
        duration_secs = int((size_kb / 192) * 8)

        return jsonify({
            "status":    "ok",
            "filename":  f"{filename}.mp3",
            "num_notes": num_notes,
            "temperature": temperature,
            "duration":  duration_secs
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/songs")
def list_songs():
    """Returns a list of all generated MP3s with metadata."""
    mp3_files = sorted(glob.glob(os.path.join(GENERATED_DIR, "*.mp3")), reverse=True)
    songs = []

    for path in mp3_files:
        fname     = os.path.basename(path)
        size_kb   = os.path.getsize(path) / 1024
        duration  = int((size_kb / 192) * 8)
        songs.append({
            "filename": fname,
            "duration": duration,
            "size_kb":  round(size_kb)
        })

    return jsonify(songs)


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serves an MP3 file to the browser audio player."""
    path = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, mimetype="audio/mpeg")


if __name__ == "__main__":
    os.makedirs(GENERATED_DIR, exist_ok=True)
    print("\nTuneChange is running at: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)