"""
Custom Keras layers for the Transformer model.
Includes PositionalEncoding and TransformerBlock layers.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    LayerNormalization,
    Dropout,
    MultiHeadAttention,
    Dense,
)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding layer for Transformer architecture.
    Adds positional information to input embeddings.
    """

    def __init__(self, position, d_model, **kwargs):
        """
        Initialize positional encoding layer.

        Args:
            position: Maximum sequence length
            d_model: Dimension of the model (embedding dimension)
            **kwargs: Additional keras layer arguments
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "position": self.position,
                "d_model": self.d_model,
            }
        )
        return config

    def get_angles(self, position, i, d_model):
        """
        Calculate angles for positional encoding.

        Args:
            position: Position indices
            i: Dimension indices
            d_model: Model dimension

        Returns:
            Angle values for encoding
        """
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        """
        Generate positional encoding matrix.

        Args:
            position: Maximum sequence length
            d_model: Model dimension

        Returns:
            Positional encoding tensor of shape (1, position, d_model)
        """
        # Get angle rates
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model,
        )

        # Apply sin to even indices (2i)
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices (2i+1)
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Interleave sines and cosines
        pos_encoding = tf.concat([sines, cosines], axis=-1)

        # Add batch dimension
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        """
        Add positional encoding to inputs.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]


class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer block with multi-head attention and feed-forward network.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3, **kwargs):
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward network
            rate: Dropout rate
            **kwargs: Additional keras layer arguments
        """
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # Multi-head attention layer
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        # Feed-forward network
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )

        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config

    def call(self, inputs, training=None):
        """
        Forward pass through transformer block.

        Args:
            inputs: Input tensor
            training: Boolean indicating training mode

        Returns:
            Transformed tensor
        """
        # Multi-head attention with residual connection
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


if __name__ == "__main__":
    print("Custom Layers Module")
    print("=" * 70)
    print("Available custom layers:")
    print("  - PositionalEncoding: Adds positional information to embeddings")
    print("  - TransformerBlock: Complete transformer block with attention + FFN")
    print("\nThese layers are used to build the transformer model architecture.")
