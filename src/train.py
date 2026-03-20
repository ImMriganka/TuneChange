import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from src.model import build_model


def load_data(processed_dir: str = "data/processed"):
    """Load the preprocessed arrays and mappings from disk."""
    print("Loading preprocessed data...")

    X = np.load(os.path.join(processed_dir, "X.npy"))
    y = np.load(os.path.join(processed_dir, "y.npy"))

    with open(os.path.join(processed_dir, "int_to_note.pkl"), "rb") as f:
        int_to_note = pickle.load(f)
    with open(os.path.join(processed_dir, "note_to_int.pkl"), "rb") as f:
        note_to_int = pickle.load(f)

    print(f"  X shape:     {X.shape}")
    print(f"  y shape:     {y.shape}")
    print(f"  Vocab size:  {len(int_to_note)}")

    return X, y, note_to_int, int_to_note


def train(
    epochs: int        = 50,
    batch_size: int    = 128,
    sequence_length: int = 64,
    processed_dir: str = "data/processed",
    checkpoint_dir: str = "outputs"
):
    # ── Load data ─────────────────────────────────────────────────────────
    X, y, note_to_int, int_to_note = load_data(processed_dir)
    vocab_size = len(int_to_note)

    # ── Build model ───────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(sequence_length, vocab_size)

    # ── Callbacks ─────────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")

    callbacks = [

        # Saves the model automatically whenever val_loss improves
        # WHY: training can take hours — this ensures you never lose
        # your best weights even if you stop early or it crashes
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        # Stops training if val_loss hasn't improved for 8 epochs
        # WHY: prevents wasting time once the model has converged
        # restore_best_weights=True rolls back to the best checkpoint
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        # Halves the learning rate if val_loss plateaus for 4 epochs
        # WHY: when loss stops improving, smaller steps help the model
        # find a better minimum instead of bouncing around
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nStarting training...")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Training samples:{X.shape[0]}")
    print(f"  Vocab size:      {vocab_size}")
    print(f"  Checkpoint:      {checkpoint_path}\n")

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,    # 10% of data used for validation
        callbacks=callbacks,
        verbose=1
    )

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "final_model.keras")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    print(f"Best model saved to:  {checkpoint_path}")

    # ── Training summary ──────────────────────────────────────────────────
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_loss  = min(history.history["val_loss"])
    print(f"\nBest epoch:     {best_epoch}")
    print(f"Best val_loss:  {best_loss:.4f}")

    return model, history


if __name__ == "__main__":
    train(
        epochs=50,
        batch_size=128,
        sequence_length=64
    )
    
