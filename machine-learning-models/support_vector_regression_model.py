import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt

# Veri Yükle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
series = pd.read_csv(url, header=0, index_col=0, parse_dates=True).asfreq('MS').ffill()

df = series.copy()
for lag in range(1, 13):
    df[f'lag_{lag}'] = df['Passengers'].shift(lag)
df = df.dropna()

train, test = df[:-12], df[-12:]

X_train, y_train = train.drop(columns=['Passengers']), train['Passengers']
X_test, y_test = test.drop(columns=['Passengers']), test['Passengers']

model = SVR()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"SVR RMSE: {rmse:.3f}")
print("Tahmin Edilen Değerler:")
print(y_pred)

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Gerçek Değerler', color='black')
plt.plot(y_pred, label=f'SVR Tahminleri (RMSE={rmse:.2f})', linestyle='--', color='orange')
plt.legend()
plt.grid(True)
plt.show()