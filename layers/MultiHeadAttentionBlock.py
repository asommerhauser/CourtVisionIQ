from keras import layers, models

class MultiHeadAttentionBlock(layers.Layer):
    def __init__(self, d_model, num_heads=4, ff_dim=256, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ff = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, X, Y, training=None):
        attn = self.mha(query=X, value=Y, key=Y, training=training)
        H = self.ln1(X + attn)
        ff_out = self.ff(H, training=training)
        return self.ln2(H + ff_out)
