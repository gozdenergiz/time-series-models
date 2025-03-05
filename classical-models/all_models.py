import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt

# Veri Setini Yükleme
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
series = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
series.index.name = 'date'
series = series.asfreq('D').ffill()

# Eğitim ve Test Setine Ayırma (Son 365 gün test olarak ayrılıyor)
train = series[:-365].copy()
test = series[-365:].copy()
train_series = train['Temp']
test_series = test['Temp']

# AR Modeli
ar_model = ARIMA(train_series, order=(7,0,0)).fit()
ar_forecast = ar_model.forecast(steps=len(test_series))
ar_rmse = sqrt(mean_squared_error(test_series, ar_forecast))

# MA Modeli
ma_model = ARIMA(train_series, order=(0,0,5)).fit()
ma_forecast = ma_model.forecast(steps=len(test_series))
ma_rmse = sqrt(mean_squared_error(test_series, ma_forecast))

# ARMA Modeli
arma_model = ARIMA(train_series, order=(3,0,3)).fit()
arma_forecast = arma_model.forecast(steps=len(test_series))
arma_rmse = sqrt(mean_squared_error(test_series, arma_forecast))

# ARIMA Modeli
arima_model = ARIMA(train_series, order=(5,1,5)).fit()
arima_forecast = arima_model.forecast(steps=len(test_series))
arima_rmse = sqrt(mean_squared_error(test_series, arima_forecast))

# SARIMA Modeli
sarima_model = SARIMAX(train_series, order=(3,1,3), seasonal_order=(1,1,1,12)).fit()
sarima_forecast = sarima_model.forecast(steps=len(test_series))
sarima_rmse = sqrt(mean_squared_error(test_series, sarima_forecast))

# SARIMAX Modeli
exog_train = np.random.normal(0, 1, len(train_series)).reshape(-1, 1)
exog_test = np.random.normal(0, 1, len(test_series)).reshape(-1, 1)
sarimax_model = SARIMAX(train_series, order=(3,1,3), seasonal_order=(1,1,1,12), exog=exog_train).fit()
sarimax_forecast = sarimax_model.forecast(steps=len(test_series), exog=exog_test)
sarimax_rmse = sqrt(mean_squared_error(test_series, sarimax_forecast))

# Dummy değişkenleri oluştur (VAR, VARMA ve VARMAX için)
train['dummy_temp'] = train['Temp'] + np.random.normal(0, 0.5, len(train))
test['dummy_temp'] = test['Temp'] + np.random.normal(0, 0.5, len(test))

# VAR Modeli
var_model = VAR(train[['Temp', 'dummy_temp']]).fit(maxlags=7)
var_forecast = var_model.forecast(train[['Temp', 'dummy_temp']].values[-var_model.k_ar:], steps=len(test))
var_rmse = sqrt(mean_squared_error(test_series, var_forecast[:, 0]))

# VARMA Modeli
varma_model = VARMAX(train[['Temp', 'dummy_temp']], order=(1, 1), trend='c')
varma_fit = varma_model.fit(disp=False)
varma_forecast = varma_fit.forecast(steps=len(test))
varma_rmse = sqrt(mean_squared_error(test_series, varma_forecast['Temp']))

# VARMAX Modeli (exog eklenmiş hali)
train['exog'] = np.random.normal(0, 1, len(train))
test['exog'] = np.random.normal(0, 1, len(test))

varmax_model = VARMAX(train[['Temp', 'dummy_temp']], order=(1, 1), trend='c', exog=train[['exog']])
varmax_fit = varmax_model.fit(disp=False)
varmax_forecast = varmax_fit.forecast(steps=len(test), exog=test[['exog']])
varmax_rmse = sqrt(mean_squared_error(test_series, varmax_forecast['Temp']))

# SES Modeli
ses_model = SimpleExpSmoothing(train_series).fit()
ses_forecast = ses_model.forecast(steps=len(test_series))
ses_rmse = sqrt(mean_squared_error(test_series, ses_forecast))

# Holt-Winters Modeli
hwes_model = ExponentialSmoothing(train_series, trend='add', seasonal='add', seasonal_periods=12).fit()
hwes_forecast = hwes_model.forecast(steps=len(test_series))
hwes_rmse = sqrt(mean_squared_error(test_series, hwes_forecast))

# RMSE Değerlerini Ekrana Yazdır
print(f'AR RMSE: {ar_rmse:.3f}')
print(f'MA RMSE: {ma_rmse:.3f}')
print(f'ARMA RMSE: {arma_rmse:.3f}')
print(f'ARIMA RMSE: {arima_rmse:.3f}')
print(f'SARIMA RMSE: {sarima_rmse:.3f}')
print(f'SARIMAX RMSE: {sarimax_rmse:.3f}')
print(f'VAR RMSE: {var_rmse:.3f}')
print(f'VARMA RMSE: {varma_rmse:.3f}')
print(f'VARMAX RMSE: {varmax_rmse:.3f}')
print(f'SES RMSE: {ses_rmse:.3f}')
print(f'HWES RMSE: {hwes_rmse:.3f}')

# Sonuçları Görselleştirme
plt.figure(figsize=(14, 8))
plt.plot(test_series, label='Gerçek Değerler', color='black')
plt.plot(test.index, ar_forecast, label=f'AR (RMSE={ar_rmse:.2f})', linestyle='--')
plt.plot(test.index, ma_forecast, label=f'MA (RMSE={ma_rmse:.2f})', linestyle='--')
plt.plot(test.index, arma_forecast, label=f'ARMA (RMSE={arma_rmse:.2f})', linestyle='--')
plt.plot(test.index, arima_forecast, label=f'ARIMA (RMSE={arima_rmse:.2f})', linestyle='--')
plt.plot(test.index, sarima_forecast, label=f'SARIMA (RMSE={sarima_rmse:.2f})', linestyle='--')
plt.plot(test.index, sarimax_forecast, label=f'SARIMAX (RMSE={sarimax_rmse:.2f})', linestyle='--')
plt.plot(test.index, var_forecast[:, 0], label=f'VAR (RMSE={var_rmse:.2f})', linestyle='--')
plt.plot(test.index, varma_forecast['Temp'], label=f'VARMA (RMSE={varma_rmse:.2f})', linestyle='--')
plt.plot(test.index, varmax_forecast['Temp'], label=f'VARMAX (RMSE={varmax_rmse:.2f})', linestyle='--')
plt.plot(test.index, ses_forecast, label=f'SES (RMSE={ses_rmse:.2f})', linestyle='--')
plt.plot(test.index, hwes_forecast, label=f'HWES (RMSE={hwes_rmse:.2f})', linestyle='--')

plt.legend()
plt.title('Classical Time Series Models - Forecasting All Test Set at Once')
plt.grid(True)
plt.show()