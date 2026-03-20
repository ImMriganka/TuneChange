# TuneChange 🎵

An AI music generation project that uses a two-layer LSTM neural network
to learn musical patterns from classical MIDI files and generate original
instrumental compositions.

## What it does
- Trains on 2,500+ classical MIDI files (Bach, Beethoven, Mozart, Nottingham dataset)
- Learns note sequences using an LSTM with 850K parameters
- Generates original melodies as MIDI files
- Converts generated MIDI to MP3 using FluidSynth
- Includes a local web UI (Flask) to generate and play songs in the browser

## Tech stack
- Python 3.10
- TensorFlow / Keras (LSTM model)
- music21 (MIDI parsing)
- NumPy / Pandas (data processing)
- Flask (web UI)
- FluidSynth + ffmpeg (audio conversion)

## Project structure
```
TuneChange/
├── data/raw/          # MIDI files (not included — see setup)
├── data/processed/    # Preprocessed arrays (generated locally)
├── outputs/           # Trained models and generated songs
├── src/
│   ├── preprocess.py  # MIDI → NumPy arrays
│   ├── model.py       # LSTM architecture
│   ├── train.py       # Training loop
│   └── generate.py    # Music generation + MP3 conversion
├── templates/
│   └── index.html     # Web UI
├── app.py             # Flask server
└── requirements.txt
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/TuneChange.git
cd TuneChange
```

### 2. Create conda environment
```bash
conda create -n tunechange python=3.10
conda activate tunechange
pip install -r requirements.txt
```

### 3. Get MIDI data
```bash
python - <<'EOF'
from music21 import corpus
import shutil, os
composers = ['bach', 'beethoven', 'mozart', 'handel', 'haydn']
os.makedirs('data/raw', exist_ok=True)
for c in composers:
    for p in corpus.getComposer(c):
        shutil.copy(str(p), os.path.join('data/raw', os.path.basename(str(p))))
EOF
```

### 4. Preprocess
```bash
python src/preprocess.py
```

### 5. Train
```bash
python -m src.train
```

### 6. Launch the UI
```bash
python app.py
# Open http://127.0.0.1:5000
```

## Results
- Training: 22 epochs, best val_loss 3.79
- Vocabulary: 237 unique notes/chords
- Training samples: 465,285 sequences
- Hardware: Apple M4 Pro (Metal GPU)