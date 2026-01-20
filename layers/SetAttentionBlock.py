from keras import layers
from .MultiHeadAttentionBlock import MultiHeadAttentionBlock

class SetAttentionBlock(layers.Layer):
    def __init__(self, d_model, num_heads=4, ff_dim=256, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mab = MultiHeadAttentionBlock(d_model, num_heads, ff_dim, dropout)

    def call(self, X, training=None):
        return self.mab(X, X, training=training)