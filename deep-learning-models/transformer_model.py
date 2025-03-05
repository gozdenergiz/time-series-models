import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from data_prep import prepare_air_quality_data

X_train, y_train, X_test, y_test, scaler = prepare_air_quality_data()

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

model = tf.keras.Sequential([
    InputLayer(input_shape=(X_train.shape[1], 1)),
    TransformerBlock(embed_dim=1, num_heads=2, ff_dim=32),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

y_pred = model.predict(X_test).flatten()
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(f"Transformer Model RMSE: {rmse:.3f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Gerçek Değerler', color='black')
plt.plot(y_pred, label=f'Transformer Tahminleri (RMSE={rmse:.3f})', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()