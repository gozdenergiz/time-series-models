import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Veri Setini Yükle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
series = pd.read_csv(url, header=0, index_col=0, parse_dates=True).asfreq('MS').ffill()

# Lag Features (Son 12 ayı lag olarak kullan)
df = series.copy()
for lag in range(1, 13):
    df[f'lag_{lag}'] = df['Passengers'].shift(lag)

df = df.dropna()

# Train/Test Ayrımı (Son 12 ay test)
train = df[:-12]
test = df[-12:]

X_train = train.drop(columns=['Passengers'])
y_train = train['Passengers']

X_test = test.drop(columns=['Passengers'])
y_test = test['Passengers']

# Modeller ve Sonuçlar
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(),
    'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
    'LightGBM': LGBMRegressor(verbose=-1),
    'CatBoost': CatBoostRegressor(verbose=0, random_seed=42)
}

results = {}

# Tüm modelleri eğit ve test et
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'model': model, 'rmse': rmse, 'y_pred': y_pred}
    print(f"{name} RMSE: {rmse:.3f}")

# Tüm tahminleri grafikte göster
plt.figure(figsize=(14, 8))
plt.plot(y_test.values, label='Gerçek Değerler', color='black')

for name, result in results.items():
    plt.plot(result['y_pred'], label=f'{name} (RMSE={result["rmse"]:.2f})', linestyle='--')

plt.legend()
plt.title('Machine Learning-Based Time Series Forecasting - AirPassengers')
plt.grid(True)
plt.show()
