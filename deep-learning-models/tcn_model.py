import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, InputLayer, BatchNormalization, ReLU, Flatten
from sklearn.metrics import mean_squared_error
from math import sqrt
from data_prep import prepare_air_quality_data

X_train, y_train, X_test, y_test, scaler = prepare_air_quality_data()

model = Sequential([
    InputLayer(input_shape=(X_train.shape[1], 1)),
    Conv1D(32, kernel_size=2, dilation_rate=1, padding='causal'),
    BatchNormalization(),
    ReLU(),
    Conv1D(32, kernel_size=2, dilation_rate=2, padding='causal'),
    BatchNormalization(),
    ReLU(),
    Flatten(),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

y_pred = model.predict(X_test).flatten()
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(f"TCN Model RMSE: {rmse:.3f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Gerçek Değerler', color='black')
plt.plot(y_pred, label=f'TCN Tahminleri (RMSE={rmse:.3f})', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()