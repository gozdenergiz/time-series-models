import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from math import sqrt

# Veri setini al ve hazırla
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
series = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
series = series.asfreq('D').ffill()

# Ek bir sahte değişken ekleyelim (normalde bu dışsal bir veri olabilir)
series['dummy_temp'] = series['Temp'] + np.random.normal(0, 0.5, len(series))

# Eğitim ve test setlerini ayır
train = series[:-365]
test = series[-365:]

# VAR modeli çok değişkenli veri ister, bu yüzden hem Temp hem dummy_temp kullanacağız
train_data = train[['Temp', 'dummy_temp']]
test_data = test[['Temp', 'dummy_temp']]

# VAR Modelini eğit
model = VAR(train_data)
model_fit = model.fit(maxlags=7)

# Test seti kadar ileriye tahmin yap
lag_order = model_fit.k_ar
forecast_input = train_data.values[-lag_order:]

forecast = model_fit.forecast(forecast_input, steps=len(test))

# Sadece Temp kolonunun tahminini kullanacağız
forecast_temp = forecast[:, 0]

# RMSE hesapla
rmse = sqrt(mean_squared_error(test['Temp'], forecast_temp))
print(f"VAR Model RMSE: {rmse:.3f}")
print("Tahmin Edilen Değerler:")
print(forecast_temp)

# Sonuçları görselleştir
plt.figure(figsize=(12, 6))
plt.plot(test['Temp'], label="Gerçek Değerler", color="black")
plt.plot(test.index, forecast_temp, label=f"VAR Tahminleri (RMSE={rmse:.2f})", linestyle="--", color="darkblue")
plt.legend()
plt.title("VAR Model - Forecast vs Actual")
plt.grid(True)
plt.show()
