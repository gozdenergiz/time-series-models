import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt

# Veri setini al ve hazırla
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
series = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
series = series.asfreq('D').ffill()

# Eğitim ve test setlerini ayır
train = series[:-365]
test = series[-365:]

train_series = train['Temp']
test_series = test['Temp']

# SARIMA Modeli (p=3, d=1, q=3), (P=1, D=1, Q=1, s=12)
model = SARIMAX(train_series, 
                order=(3,1,3), 
                seasonal_order=(1,1,1,12)).fit()

# Test seti kadar ileriye dönük tahmin yap
forecast = model.forecast(steps=len(test_series))

# RMSE hesapla
rmse = sqrt(mean_squared_error(test_series, forecast))
print(f"SARIMA Model RMSE: {rmse:.3f}")
print("Tahmin Edilen Değerler:")
print(forecast.values)

# Sonuçları görselleştir
plt.figure(figsize=(12, 6))
plt.plot(test_series, label="Gerçek Değerler", color="black")
plt.plot(test.index, forecast, label=f"SARIMA Tahminleri (RMSE={rmse:.2f})", linestyle="--", color="darkgreen")
plt.legend()
plt.title("SARIMA Model - Forecast vs Actual")
plt.grid(True)
plt.show()