"""
Improved Transformer model architecture for sign language detection.
Optimized for >90% accuracy with faster training.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    LayerNormalization,
    BatchNormalization,
    Concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from src.models.custom_layers import PositionalEncoding, TransformerBlock
from src.config import (
    SEQUENCE_LENGTH,
    TOTAL_KEYPOINTS,
    EMBED_DIM,
    NUM_HEADS,
    FF_DIM,
    DROPOUT_RATE,
    LEARNING_RATE,
    get_num_classes,
)


def create_transformer_model(
    num_actions=None,
    sequence_length=None,
    input_dim=None,
    embed_dim=None,
    num_heads=None,
    ff_dim=None,
    dropout_rate=None,
    learning_rate=None,
):
    """
    Create improved transformer model with better accuracy.

    Key improvements:
    - Dual transformer blocks for deeper learning
    - Batch normalization for stable training
    - L2 regularization to prevent overfitting
    - Residual connections for gradient flow
    - Larger dense layers before output

    Args:
        num_actions: Number of output classes (default: from config)
        sequence_length: Length of input sequences (default: from config)
        input_dim: Dimension of input features (default: from config)
        embed_dim: Embedding dimension (default: from config)
        num_heads: Number of attention heads (default: from config)
        ff_dim: Feed-forward dimension (default: from config)
        dropout_rate: Dropout rate (default: from config)
        learning_rate: Learning rate for optimizer (default: from config)

    Returns:
        Compiled Keras model
    """
    # Use config defaults if not provided
    num_actions = num_actions or get_num_classes()
    sequence_length = sequence_length or SEQUENCE_LENGTH
    input_dim = input_dim or TOTAL_KEYPOINTS
    embed_dim = embed_dim or EMBED_DIM
    num_heads = num_heads or NUM_HEADS
    ff_dim = ff_dim or FF_DIM
    dropout_rate = dropout_rate or DROPOUT_RATE
    learning_rate = learning_rate or LEARNING_RATE

    print("=" * 70)
    print("CREATING IMPROVED TRANSFORMER MODEL")
    print("=" * 70)
    print(f"Input shape: ({sequence_length}, {input_dim})")
    print(f"Output classes: {num_actions}")
    print(f"Embed dim: {embed_dim}")
    print(f"Attention heads: {num_heads}")
    print(f"Feed-forward dim: {ff_dim}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 70)

    # Input layer
    inputs = Input(shape=(sequence_length, input_dim), name="input_sequences")

    # Project input to embedding dimension with batch norm
    x = Dense(
        embed_dim, kernel_regularizer=regularizers.l2(1e-4), name="input_projection"
    )(inputs)
    x = BatchNormalization(name="bn_projection")(x)

    # Add positional encoding
    x = PositionalEncoding(sequence_length, embed_dim, name="positional_encoding")(x)

    # First Transformer block
    x1 = TransformerBlock(
        embed_dim, num_heads, ff_dim, rate=dropout_rate, name="transformer_block_1"
    )(x)

    # Second Transformer block for deeper learning
    x2 = TransformerBlock(
        embed_dim, num_heads, ff_dim, rate=dropout_rate, name="transformer_block_2"
    )(x1)

    # Add residual connection from first block
    x_combined = tf.keras.layers.Add(name="residual_connection")([x1, x2])
    x_combined = LayerNormalization(name="final_norm")(x_combined)

    # Global average pooling
    x_pooled = GlobalAveragePooling1D(name="global_avg_pooling")(x_combined)

    # Enhanced dense layers with batch normalization
    x = Dense(
        256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_1"
    )(x_pooled)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)

    x = Dense(
        128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_2"
    )(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(0.3, name="dropout_2")(x)

    x = Dense(
        64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_3"
    )(x)
    x = Dropout(0.2, name="dropout_3")(x)

    # Output layer
    outputs = Dense(num_actions, activation="softmax", name="output")(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="improved_sign_transformer")

    # Compile with optimized settings
    optimizer = Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model


def load_trained_model(model_path):
    """
    Load a pre-trained model with custom layers.

    Args:
        model_path: Path to saved model file (.h5 or .keras)

    Returns:
        Loaded Keras model
    """
    from tensorflow.keras.models import load_model

    # Define custom objects for loading
    custom_objects = {
        "PositionalEncoding": PositionalEncoding,
        "TransformerBlock": TransformerBlock,
    }

    print(f"Loading model from: {model_path}")
    model = load_model(model_path, custom_objects=custom_objects)
    print("✓ Model loaded successfully!")

    return model


if __name__ == "__main__":
    print("Improved Transformer Model Module")
    print("=" * 70)

    # Test model creation
    print("\nTesting model creation...")
    try:
        model = create_transformer_model()
        print("\n✓ Model created successfully!")
        print(f"\nModel summary:")
        model.summary()

        print(f"\nTotal parameters: {model.count_params():,}")

    except Exception as e:
        print(f"\n✗ Error creating model: {e}")
        import traceback

        traceback.print_exc()
