import numpy as np
import pandas as pd
from pandas import DataFrame, Series, concat, read_csv
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Input
from matplotlib import pyplot as plt
from math import sqrt

# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row).reshape(1, -1)
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an RNN model to training data
def fit_rnn(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))  # shape=(timesteps, features)
    model.add(SimpleRNN(neurons))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1)
    return model

# make a one-step forecast
def forecast_rnn(model, X):
    X = X.reshape(1, X.shape[0], 1)
    yhat = model.predict(X, verbose=0)
    return yhat[0, 0]

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
series = read_csv(url, header=0, parse_dates=[0], index_col=0, date_parser=parser)

# transform data to be stationary (differencing)
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split into train and test sets
train, test = supervised_values[0:-12], supervised_values[-12:]

# scale the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit RNN
rnn_model = fit_rnn(train_scaled, batch_size=1, nb_epoch=200, neurons=4)

# one-step forecast için state hazırlığı
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
rnn_model.predict(train_reshaped, verbose=0)

# walk-forward validation
predictions = []
for i in range(len(test_scaled)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_rnn(rnn_model, X)
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# calculate RMSE
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print(f'Test RMSE: {rmse:.3f}')

# plot predicted vs actual
plt.plot(raw_values[-12:], label='Gerçek Değerler')
plt.plot(predictions, label='RNN Tahminleri', linestyle='--')
plt.legend()
plt.show()
