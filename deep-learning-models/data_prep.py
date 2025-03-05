import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_air_quality_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.csv"
    df = pd.read_csv(url, sep=';', decimal=',', parse_dates=[['Date', 'Time']], na_values=-200)

    df = df[['Date_Time', 'AH']]
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    df.set_index('Date_Time', inplace=True)
    df = df.ffill().dropna()

    scaler = MinMaxScaler()
    df['AH'] = scaler.fit_transform(df[['AH']])

    for lag in range(1, 25):
        df[f'lag_{lag}'] = df['AH'].shift(lag)

    df = df.dropna()
    train = df[:-24*30]
    test = df[-24*30:]

    X_train = train.drop(columns=['AH'])
    y_train = train['AH']
    X_test = test.drop(columns=['AH'])
    y_test = test['AH']

    X_train_seq = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_seq = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train_seq, y_train, X_test_seq, y_test, scaler
