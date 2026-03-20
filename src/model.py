import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_model(sequence_length: int, vocab_size: int) -> tf.keras.Model:
    """
    Builds a two-layer LSTM model for music generation.

    Args:
        sequence_length: number of timesteps in each input (64 in our case)
        vocab_size:       number of unique notes/chords (237 in our case)

    Returns:
        A compiled Keras model ready for training
    """

    model = Sequential([

        # ── Explicit input layer ──────────────────────────────────────────
        # Keras 3 prefers this over passing input_shape to the LSTM directly
        # shape = (timesteps, features) = (64, 1)
        Input(shape=(sequence_length, 1)),

        # ── Layer 1: First LSTM ───────────────────────────────────────────
        # 256 units = 256 memory cells learning musical patterns
        # return_sequences=True → passes all 64 timesteps to next LSTM layer
        LSTM(
            units=256,
            return_sequences=True
        ),

        # ── Layer 2: Dropout ──────────────────────────────────────────────
        # Drops 30% of neurons randomly during each training step
        # WHY: prevents the model from memorising specific songs
        # instead of learning general musical patterns
        Dropout(0.3),

        # ── Layer 3: Second LSTM ──────────────────────────────────────────
        # return_sequences=False → only outputs the LAST timestep
        # This collapses (batch, 64, 256) → (batch, 256)
        # The model has now "read" the full sequence and summarised it
        LSTM(
            units=256,
            return_sequences=False
        ),

        # ── Layer 4: Dropout ──────────────────────────────────────────────
        Dropout(0.3),

        # ── Layer 5: Dense output ─────────────────────────────────────────
        # 237 neurons = one per note in our vocabulary
        # softmax turns raw scores into probabilities that sum to 1.0
        Dense(
            units=vocab_size,
            activation='softmax'
        )

    ])

    # ── Compile ───────────────────────────────────────────────────────────
    # sparse_categorical_crossentropy: used when labels are integers (our y)
    # Adam lr=0.001: standard starting point, adjusts itself during training
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    SEQUENCE_LENGTH = 64
    VOCAB_SIZE      = 237

    model = build_model(SEQUENCE_LENGTH, VOCAB_SIZE)
    model.summary()