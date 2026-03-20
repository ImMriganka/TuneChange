import os
import glob
import numpy as np
import pandas as pd
import pickle
from music21 import converter, instrument, note, chord
from tqdm import tqdm


def parse_midi_files(midi_folder: str) -> list:
    all_notes = []

    patterns = ["**/*.mid", "**/*.midi", "**/*.mxl", "**/*.xml"]
    midi_paths = []
    for pattern in patterns:
        midi_paths += glob.glob(os.path.join(midi_folder, pattern), recursive=True)

    if not midi_paths:
        raise FileNotFoundError(f"No music files found in: {midi_folder}")

    print(f"Found {len(midi_paths)} music files")

    for path in tqdm(midi_paths, desc="Parsing files"):
        try:
            midi = converter.parse(path)
            try:
                parts = instrument.partitionByInstrument(midi)
                notes_to_parse = parts.parts[0].recurse()
            except Exception:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    all_notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    all_notes.append(
                        ".".join(str(n) for n in sorted(element.normalOrder))
                    )

        except Exception as e:
            print(f"\nWARNING: Could not parse {os.path.basename(path)}: {e}")
            continue

    print(f"\nTotal notes/chords extracted: {len(all_notes)}")
    return all_notes


def build_vocabulary(notes: list):
    notes_series = pd.Series(notes)
    unique_notes = sorted(notes_series.unique())
    vocab_size = len(unique_notes)
    print(f"Vocabulary size (unique notes/chords): {vocab_size}")

    note_to_int = {n: idx for idx, n in enumerate(unique_notes)}
    int_to_note = {idx: n for idx, n in enumerate(unique_notes)}

    vocab_df = pd.DataFrame({
        "note":      unique_notes,
        "integer":   range(vocab_size),
        "frequency": [notes_series.value_counts()[n] for n in unique_notes]
    }).sort_values("frequency", ascending=False).reset_index(drop=True)

    return note_to_int, int_to_note, vocab_df


def create_sequences(notes: list, note_to_int: dict, sequence_length: int = 64):
    vocab_size = len(note_to_int)
    encoded = np.array([note_to_int[n] for n in notes], dtype=np.int32)

    inputs, outputs = [], []

    for i in tqdm(range(len(encoded) - sequence_length), desc="Building sequences"):
        inputs.append(encoded[i : i + sequence_length])
        outputs.append(encoded[i + sequence_length])

    X = np.array(inputs,  dtype=np.float32)
    y = np.array(outputs, dtype=np.int32)

    # Reshape to (samples, timesteps, features) — required by Keras LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Normalise to [0, 1]
    X = X / float(vocab_size)

    print(f"\nSequence shapes:")
    print(f"  X: {X.shape}  →  (samples, timesteps, features)")
    print(f"  y: {y.shape}  →  (samples,)")

    return X, y


def save_processed_data(X, y, note_to_int, int_to_note, notes,
                        output_dir: str = "data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving arrays to disk...")
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    with open(os.path.join(output_dir, "note_to_int.pkl"), "wb") as f:
        pickle.dump(note_to_int, f)
    with open(os.path.join(output_dir, "int_to_note.pkl"), "wb") as f:
        pickle.dump(int_to_note, f)
    with open(os.path.join(output_dir, "notes.pkl"), "wb") as f:
        pickle.dump(notes, f)

    print(f"All processed data saved to '{output_dir}/'")
    print(f"\nFiles in data/processed/:")
    for f in os.listdir(output_dir):
        size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
        print(f"  {f:25} {size_mb:.1f} MB")


if __name__ == "__main__":
    MIDI_FOLDER     = "data/raw"
    SEQUENCE_LENGTH = 64

    print("── Step 1: Parsing files ──")
    notes = parse_midi_files(MIDI_FOLDER)

    print("\n── Step 2: Building vocabulary ──")
    note_to_int, int_to_note, vocab_df = build_vocabulary(notes)
    print("\nTop 10 most frequent notes:")
    print(vocab_df.head(10).to_string(index=False))

    print("\n── Step 3: Creating sequences ──")
    X, y = create_sequences(notes, note_to_int, SEQUENCE_LENGTH)

    print("\n── Step 4: Saving to disk ──")
    save_processed_data(X, y, note_to_int, int_to_note, notes)