import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VARMAX
from sklearn.metrics import mean_squared_error
from math import sqrt

# Veri setini al ve hazırla
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
series = pd.read_csv(url, header=0, index_col=0, parse_dates=True).asfreq('D').ffill()

# Durağanlaştırma (fark alma)
series['Temp_diff'] = series['Temp'].diff().fillna(0)

# Ek bir sahte değişken ekleyelim (VARMA multivariate olduğu için 2 değişken lazım)
series['dummy_temp'] = series['Temp_diff'] + np.random.normal(0, 0.5, len(series))

# Eğitim ve test setlerini ayır
train = series[:-365]
test = series[-365:]

train_data = train[['Temp_diff', 'dummy_temp']]
test_data = test[['Temp_diff', 'dummy_temp']]

# VARMA Modelini tanımla ve fit et (küçük order kullanıyoruz)
model = VARMAX(train_data, order=(1, 1), trend='c')
model_fit = model.fit(disp=False)

# Test seti kadar ileriye tahmin yap
forecast = model_fit.forecast(steps=len(test))

# Sadece Temp_diff kolonunun tahminini kullanacağız
forecast_diff = forecast['Temp_diff']

# Farklı değerleri orijinal forma dönüştür (cumsum ile farkı geri al)
forecast_temp = series['Temp'].iloc[-366] + np.cumsum(forecast_diff)

# RMSE hesapla (gerçek temp ile karşılaştır)
rmse = sqrt(mean_squared_error(test['Temp'], forecast_temp))
print(f"VARMA Model RMSE: {rmse:.3f}")
print("Tahmin Edilen Değerler:")
print(forecast_temp.values)

# Sonuçları görselleştir
plt.figure(figsize=(12, 6))
plt.plot(test['Temp'], label="Gerçek Değerler", color="black")
plt.plot(test.index, forecast_temp, label=f"VARMA Tahminleri (RMSE={rmse:.2f})", linestyle="--", color="teal")
plt.legend()
plt.title("VARMA Model - Forecast vs Actual")
plt.grid(True)
plt.show()
