import os
import sys
import numpy as np
import pickle
import random
import tensorflow as tf
from music21 import stream, note, chord, instrument


def load_model_and_data(
    model_path: str    = "outputs/best_model.keras",
    processed_dir: str = "data/processed"
):
    """Load the trained model and vocabulary mappings."""
    print("Loading model and data...")

    model = tf.keras.models.load_model(model_path)

    with open(os.path.join(processed_dir, "int_to_note.pkl"), "rb") as f:
        int_to_note = pickle.load(f)
    with open(os.path.join(processed_dir, "note_to_int.pkl"), "rb") as f:
        note_to_int = pickle.load(f)
    with open(os.path.join(processed_dir, "notes.pkl"), "rb") as f:
        notes = pickle.load(f)

    vocab_size = len(int_to_note)
    print(f"  Model loaded:   {model_path}")
    print(f"  Vocab size:     {vocab_size}")
    print(f"  Notes in pool:  {len(notes)}")

    return model, notes, note_to_int, int_to_note, vocab_size


def pick_seed(notes: list, note_to_int: dict, sequence_length: int = 64):
    """
    Pick a random 64-note window from the real training data as the seed.
    WHY: starting from real musical context gives the model a better
    launchpad than random notes, producing more coherent output.
    """
    start_idx = random.randint(0, len(notes) - sequence_length - 1)
    seed_notes = notes[start_idx : start_idx + sequence_length]
    seed_ints  = [note_to_int[n] for n in seed_notes]

    print(f"\nSeed sequence (first 10 notes): {seed_notes[:10]}")
    return seed_ints


def sample_with_temperature(predictions: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample from the model's probability distribution with temperature control.

    WHY temperature matters:
      temperature < 1.0  → more conservative, picks high-probability notes
                           sounds more repetitive but stays "in key"
      temperature = 1.0  → uses raw model probabilities (balanced)
      temperature > 1.0  → more adventurous, explores lower-probability notes
                           sounds more varied but can get chaotic

    Args:
        predictions: softmax output array of shape (vocab_size,)
        temperature: float controlling randomness

    Returns:
        integer index of the sampled note
    """
    predictions = np.asarray(predictions).astype("float64")

    # Divide log-probabilities by temperature then re-normalise
    # WHY log: more numerically stable than dividing raw probabilities
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds   = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    # np.random.multinomial draws one sample from the distribution
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)


def generate_notes(
    model,
    seed_ints:       list,
    note_to_int:     dict,
    int_to_note:     dict,
    vocab_size:      int,
    num_notes:       int   = 200,
    sequence_length: int   = 64,
    temperature:     float = 0.8
) -> list:
    """
    Autoregressively generate a sequence of notes.

    The loop:
      1. Take the current 64-note window
      2. Normalise and reshape to (1, 64, 1)
      3. Model predicts probability over 237 notes
      4. Sample one note using temperature
      5. Append it, slide the window forward by 1
      6. Repeat num_notes times
    """
    print(f"\nGenerating {num_notes} notes (temperature={temperature})...")

    # Start with a copy of the seed — we'll slide this window forward
    current_sequence = list(seed_ints)
    generated        = []

    for i in range(num_notes):
        # Take the last sequence_length notes
        window = current_sequence[-sequence_length:]

        # Reshape to (1, 64, 1) and normalise — same as training
        X = np.array(window, dtype=np.float32)
        X = X.reshape(1, sequence_length, 1)
        X = X / float(vocab_size)

        # Get probability distribution over all 237 notes
        predictions = model.predict(X, verbose=0)[0]

        # Sample one note index using temperature
        next_int = sample_with_temperature(predictions, temperature)

        # Decode back to note string and store
        generated.append(int_to_note[next_int])

        # Slide the window: drop oldest note, add new prediction
        current_sequence.append(next_int)

        # Progress indicator every 50 notes
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{num_notes} notes...")

    print(f"Generation complete. Sample: {generated[:8]}")
    return generated


def notes_to_midi(
    generated_notes: list,
    output_path:     str   = "outputs/generated/song.mid",
    tempo_bpm:       int   = 120
):
    """
    Convert the list of note/chord strings back into a playable MIDI file.

    Each string is either:
      - A single note like 'C4', 'F#3'     → music21 note.Note
      - A chord like '0.4.7'               → music21 chord.Chord
    """
    print(f"\nConverting {len(generated_notes)} notes to MIDI...")

    output_stream = stream.Stream()
    output_stream.append(instrument.Piano())

    # Each note gets a fixed duration of 0.5 quarter notes
    # WHY 0.5: gives a natural 8th-note feel typical of classical pieces
    note_duration = 0.5

    for pattern in generated_notes:
        if "." in pattern or pattern.isdigit():
            # It's a chord — split by dot and build note objects
            chord_notes = pattern.split(".")
            notes_in_chord = []

            for n in chord_notes:
                try:
                    new_note = note.Note(int(n))
                    new_note.storedInstrument = instrument.Piano()
                    notes_in_chord.append(new_note)
                except Exception:
                    pass

            if notes_in_chord:
                new_chord = chord.Chord(notes_in_chord)
                new_chord.duration.quarterLength = note_duration
                output_stream.append(new_chord)

        else:
            # It's a single note like 'C4' or 'F#3'
            try:
                new_note = note.Note(pattern)
                new_note.storedInstrument = instrument.Piano()
                new_note.duration.quarterLength = note_duration
                output_stream.append(new_note)
            except Exception:
                pass

    # Write to MIDI file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_stream.write("midi", fp=output_path)
    print(f"MIDI saved to: {output_path}")

    return output_path


def generate_song(
    model_path:      str   = "outputs/best_model.keras",
    output_path:     str   = "outputs/generated/song.mid",
    num_notes:       int   = 200,
    temperature:     float = 0.8,
    sequence_length: int   = 64
):
    """
    Full pipeline: load model → pick seed → generate → save MIDI.
    This is the function you call to produce a new song.
    """
    # Step 1: Load everything
    model, notes, note_to_int, int_to_note, vocab_size = load_model_and_data(
        model_path=model_path
    )

    # Step 2: Pick a seed from real training data
    seed_ints = pick_seed(notes, note_to_int, sequence_length)

    # Step 3: Generate notes autoregressively
    generated = generate_notes(
        model          = model,
        seed_ints      = seed_ints,
        note_to_int    = note_to_int,
        int_to_note    = int_to_note,
        vocab_size     = vocab_size,
        num_notes      = num_notes,
        sequence_length= sequence_length,
        temperature    = temperature
    )

    # Step 4: Convert to MIDI
    midi_path = notes_to_midi(
        generated_notes = generated,
        output_path     = output_path
    )

    return midi_path

def midi_to_mp3(
    midi_path:  str,
    sf2_path:   str = "assets/FluidR3.sf3",
    output_mp3: str = None
):
    """
    Converts a MIDI file to MP3 using FluidSynth + ffmpeg.

    Args:
        midi_path:  path to your generated .mid file
        sf2_path:   path to the SoundFont (.sf2 or .sf3) file
        output_mp3: output path for the .mp3 file
                    defaults to same location as midi_path
    """
    import subprocess

    if not os.path.exists(sf2_path):
        raise FileNotFoundError(
            f"SoundFont not found at {sf2_path}\n"
            f"Download one into your assets/ folder first."
        )

    # Default output path: same folder, same name, .mp3 extension
    if output_mp3 is None:
        output_mp3 = midi_path.replace(".mid", ".mp3")

    # Intermediate WAV path
    output_wav = midi_path.replace(".mid", ".wav")

    # Step 1: MIDI → WAV using FluidSynth
    # NOTE: FluidSynth 2.4+ requires -F before the soundfont path
    print(f"\nRendering MIDI to WAV with FluidSynth...")
    subprocess.run([
        "fluidsynth",
        "-ni",                # non-interactive, no shell
        "-F", output_wav,     # output WAV file — must come before sf2
        "-r", "44100",        # 44.1kHz sample rate
        sf2_path,             # SoundFont file
        midi_path             # input MIDI file — must come last
    ], check=True)

    if not os.path.exists(output_wav):
        raise FileNotFoundError(
            f"FluidSynth did not produce a WAV file at {output_wav}\n"
            f"Check that your SoundFont file is valid."
        )

    # Step 2: WAV → MP3 using ffmpeg
    print(f"Converting WAV to MP3 with ffmpeg...")
    subprocess.run([
        "ffmpeg",
        "-y",                    # overwrite output if exists
        "-i", output_wav,        # input WAV
        "-acodec", "libmp3lame", # MP3 encoder
        "-b:a", "192k",          # 192kbps — good quality for music
        output_mp3
    ], check=True)

    # Clean up the intermediate WAV
    os.remove(output_wav)

    print(f"MP3 saved to: {output_mp3}")
    return output_mp3


if __name__ == "__main__":
    # You can tweak these three values to change the output:
    #
    # num_notes:   how long the piece is (200 = ~50 seconds at 120bpm)
    # temperature: 0.5 = safe/repetitive, 0.8 = balanced, 1.2 = adventurous
    # output_path: where the MIDI file is saved

    generate_song(
        num_notes   = 200,
        temperature = 0.8,
        output_path = "outputs/generated/song_01.mid"
    )