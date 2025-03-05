import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, InputLayer, Dropout
from sklearn.metrics import mean_squared_error
from math import sqrt
from data_prep import prepare_air_quality_data

X_train, y_train, X_test, y_test, scaler = prepare_air_quality_data()

model = Sequential([
    InputLayer(input_shape=(X_train.shape[1], 1)),
    GRU(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

y_pred = model.predict(X_test).flatten()
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(f"GRU Model RMSE: {rmse:.3f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Gerçek Değerler', color='black')
plt.plot(y_pred, label=f'GRU Tahminleri (RMSE={rmse:.3f})', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()