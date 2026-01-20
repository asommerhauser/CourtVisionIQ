from keras import layers
from .MultiHeadAttentionBlock import MultiHeadAttentionBlock

class PoolingByMultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_seeds=1, num_heads=4, ff_dim=256, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.seed = self.add_weight(
            name="seed_vectors",
            shape=(num_seeds, d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.mab = MultiHeadAttentionBlock(d_model, num_heads, ff_dim, dropout)

    def call(self, X, training=None):
        import tensorflow as tf
        B = tf.shape(X)[0]
        S = tf.tile(tf.expand_dims(self.seed, axis=0), [B, 1, 1])  # (B, k, d)
        return self.mab(S, X, training=training)  # (B, k, d)