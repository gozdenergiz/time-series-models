# from pandas import read_csv
# from matplotlib import pyplot
# series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
# print(series.head())
# series.plot()
# pyplot.show()


# create and evaluate a static autoregressive model
# from pandas import read_csv
# from matplotlib import pyplot
# from statsmodels.tsa.ar_model import AutoReg
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# # load dataset
# series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True)
# print(series.head())
# # split dataset
# X = series.values
# train, test = X[1:len(X)-7], X[len(X)-7:]
# # train autoregression
# model = AutoReg(train, lags=29)
# model_fit = model.fit()
# print('Coefficients: %s' % model_fit.params)
# # make predictions
# predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
# for i in range(len(predictions)):
# 	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
# rmse = sqrt(mean_squared_error(test, predictions))
# print('Test RMSE: %.3f' % rmse)
# # plot results
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()




# create and evaluate an updated autoregressive model
# # from pandas import read_csv
# # from matplotlib import pyplot
# # from statsmodels.tsa.ar_model import AutoReg
# # from sklearn.metrics import mean_squared_error
# # from math import sqrt
# # # load dataset
# # series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True)
# # # split dataset
# # X = series.values
# # train, test = X[1:len(X)-7], X[len(X)-7:]
# # # train autoregression
# # window = 29
# # model = AutoReg(train, lags=29)
# # model_fit = model.fit()
# # coef = model_fit.params
# # # walk forward over time steps in test
# # history = train[len(train)-window:]
# # history = [history[i] for i in range(len(history))]
# # predictions = list()
# # for t in range(len(test)):
# # 	length = len(history)
# # 	lag = [history[i] for i in range(length-window,length)]
# # 	yhat = coef[0]
# # 	for d in range(window):
# # 		yhat += coef[d+1] * lag[window-d-1]
# # 	obs = test[t]
# # 	predictions.append(yhat)
# # 	history.append(obs)
# # 	print('predicted=%f, expected=%f' % (yhat, obs))
# # rmse = sqrt(mean_squared_error(test, predictions))
# # print('Test RMSE: %.3f' % rmse)
# # # plot
# # pyplot.plot(test)
# # pyplot.plot(predictions, color='red')
# # pyplot.show()

# AR example
from statsmodels.tsa.ar_model import AutoReg
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = AutoReg(data, lags=1)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

