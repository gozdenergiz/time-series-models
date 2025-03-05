import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Veri setini yükle ve hazırla
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
series = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
series = series.asfreq('D').ffill()

# Eğitim ve test setlerine ayır
train = series[:-365]
test = series[-365:]

train_series = train['Temp']
test_series = test['Temp']

# ARMA Modeli (p=3, d=0, q=3)
model = ARIMA(train_series, order=(3,0,3)).fit()

# Test seti kadar ileriye tahmin yap
forecast = model.forecast(steps=len(test_series))

# RMSE hesapla
rmse = sqrt(mean_squared_error(test_series, forecast))
print(f"ARMA Model RMSE: {rmse:.3f}")
print("Tahmin Edilen Değerler:")
print(forecast.values)

# Sonuçları görselleştir
plt.figure(figsize=(12, 6))
plt.plot(test_series, label="Gerçek Değerler", color="black")
plt.plot(test.index, forecast, label=f"ARMA Tahminleri (RMSE={rmse:.2f})", linestyle="--", color="purple")
plt.legend()
plt.title("ARMA Model - Forecast vs Actual")
plt.grid(True)
plt.show()