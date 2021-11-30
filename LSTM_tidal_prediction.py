#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 18:57:09 2021

@author: matias
"""
"""
Created on Mon Nov  8 11:49:03 2021

I use the example given in
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

To do a forecast of the daily-averaged sea level as a function of wind speed and direction, and tidal information.

For the sea level we do a running mean of 24 hours 50 min every hour. For wind velocity and atmospheric pressure,
 we have hourly values



@author: matias
"""
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
import utide


# %% convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, n_f=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in-1, 0, -1):
		cols.append(df.loc[:,0:(n_f-1)].shift(i))
    #
	for i in range(n_in, 0, -1):    
		names += [('var%d(t-%d)' % (j+1, i-1)) for j in range(n_f)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % data.shape[1])]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_out)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# %% Load data
# Load sea level data
L = read_csv('data/level_tide.csv')


#If you want to substitute L['tide'] by a specific reconstruction
coef = utide.solve(L['time'], L['level'],
                   lat=52, 
                   method='ols',
                   conf_int='MC')

# %%
#const =  ['M2', 'S2', 'M4', 'O1', 'SA', 'N2', 'MU2', 'K1', 'MS4', 'L2', 'M6',
#         '2MS6', 'K2', 'NU2', 'MN4', 'Q1', 'P1', '2MN6', 'LDA2', 'M8', 'H1',
#         'SSA', 'MK4', 'EPS2', 'MSN2', '2MK6', 'S1', '2MK5', 'MM', 'T2',
#         '2SM6', '2N2', 'MKS2', 'MO3', 'MSK6', 'MSF', 'RHO1', 'SN4', '2Q1',
#         'H2', 'M3', 'S4', 'SO1', 'OO1', 'TAU1', 'MK3', 'SK4', 'GAM2',
#         'PSI1', 'J1', 'ETA2', 'SIG1', 'ALP1', 'PI1', 'SO3', 'BET1', '3MK7',
#         'MSM', 'CHI1', 'MF', 'R2', 'NO1', 'PHI1', 'OQ2', 'SK3', 'THE1',
#         'UPS1', '2SK5']

# Only moon
const = ['M2', 'N2', 'M4', 'O1', 'K2', 'MU2', 'K1', 'MN', 'L2']

#const  = ['M2', 'M4', 'O1', 'K1', 'MN']

# Most important
#const = ['M2', 'S2', 'M4', 'O1', 'SA', 'N2', 'MU2', 'K1', 'MS4', 'L2', 'M6', '2MS6', 'K2']

tide = utide.reconstruct(L['time'], coef, constit=const)
#tide['h'] = L['tide']
# Load astronomic data
A = pd.read_csv('data/astronomic.csv')

#%% Moon and sun azimuth into sine and cosine

gdr = np.pi/180 # useful to transform from degrees to radians

ma_cos = np.cos(A['azimuth_moon_deg']*gdr)
ma_sin = np.sin(A['azimuth_moon_deg']*gdr)
sa_cos = np.cos(A['azimuth_sun_deg']*gdr)
sa_sin = np.sin(A['azimuth_sun_deg']*gdr)
#%%
#t = (L['time']-L['time'][0])
#pyplot.plot(t[0:1000],tide['h'][0:1000],'r',label="data")
#pyplot.plot(t[0:1000],ma_cos[0:1000]*75,'b',label="distance_moon")
#pyplot.legend()
#pyplot.show()

# %% Arrange data
# FULL INPUT
#tmp = np.stack((A['altitude_moon_deg'], A['distance_moon_au'], ma_cos, ma_sin, A['altitude_sun_deg'], A['distance_sun_au'], sa_cos, sa_sin,tide['h']))
#d = {'altitude_moon_deg': tmp[0,:], 'distance_moon_au': tmp[1,:], 'azimuth_moon_cos': tmp[2,:], 'azimuth_moon_sin': tmp[3,:], 'altitude_sun_deg': tmp[4,:], 'distance_sun_au': tmp[5,:], 'azimuth_sun_cos': tmp[6,:], 'azimuth_sun_sin': tmp[7,:], 'level': tmp[8,:]}

# ONLY MOON
tmp = np.stack((A['altitude_moon_deg'], A['distance_moon_au'], ma_cos, ma_sin,tide['h']))
d = {'altitude_moon_deg': tmp[0,:], 'distance_moon_au': tmp[1,:], 'azimuth_moon_cos': tmp[2,:], 'azimuth_moon_sin': tmp[3,:], 'level': tmp[4,:]}

dataset = pd.DataFrame(data=d)
values = dataset.values

# %%
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
n_steps_in = 96  #specify the number of the previous time steps to use for the prediction = 1 in this case
n_steps_out = 1 #specify the number of time steps to predict = 1 in this case because we are predicting only 1 time step
n_features = 4 #number of features (variables) used to predict

# frame as supervised learning
reframed = series_to_supervised(scaled, n_steps_in, n_steps_out, n_features)
reframed.shape

# %%
# split into train and test sets
nsamples=reframed.shape[0] #=14107
values = reframed.values
n_train_periods = int(nsamples*0.1) #percentage for training
n_test_periods  = int(nsamples*0.1) #percentage for training
train = values[:n_train_periods, :]
test = values[n_train_periods:(n_test_periods+n_train_periods), :]
# split into input and outputs (works only with n_steps_in=n_steps_out=1)
n_obs = n_steps_in * n_features #(features=predictors) #1*3=3

#for predicting sea level at time t using predictors at time <=t
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]
#
#%%
print(train_X.shape, train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_steps_in, n_features))
test_X = test_X.reshape((test_X.shape[0], n_steps_in, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#%%
# design network
model = Sequential()
#model.add(LSTM(12, input_shape=(train_X.shape[1], train_X.shape[2]))) #=(n_steps_in,n_features)
model.add(LSTM(12, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]))) #=(n_steps_in,n_features)
model.add(LSTM(4, input_shape=(train_X.shape[1], train_X.shape[2])))
model.compile(loss='mse', optimizer='adam') #mean absolute error "mse" "mae"
# fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig("./models/loss.png", dpi=150)

# %% Save model 

model.save('./models/')

# %% Make a prediction
#yhat = model.predict(test_X)
#test_X0 = test_X.reshape((test_X.shape[0], n_steps_in*n_features))
## invert scaling for forecast
##inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
#inv_yhat = concatenate((test_X0,yhat), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat[:,-(n_features+1):])
#inv_yhat = inv_yhat[:,-1]
## invert scaling for actual
#test_y0 = test_y.reshape((len(test_y), 1))
##inv_y = concatenate((test_y, test_X[:, -4:]), axis=1)
#inv_y = concatenate((test_X0,test_y0), axis=1)
#inv_y = scaler.inverse_transform(inv_y[:,-(n_features+1):])
#inv_y = inv_y[:,-1]
## calculate RMSE
#rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#print('Test RMSE: %.3f' % rmse)
#print('Test std: %.3f' % inv_y.std())
#
##%%
#
#t = L['time'][:]-L['time'][0]
#
#pyplot.plot(inv_y, inv_yhat,'o')
#pyplot.xlabel("data")
#pyplot.ylabel("prediction")
#pyplot.grid()
#pyplot.axis([500,900,500,900])
#pyplot.axis("equal")
#pyplot.show()
#
#
#pyplot.plot(t[0:inv_y.size],inv_y,'r',label="data")
#pyplot.plot(t[0:inv_y.size],inv_yhat,'b',label="prediction")
#pyplot.legend()
#pyplot.show()
#
#pyplot.plot(t[0:600],inv_y[0:600],'r',label="data")
#pyplot.plot(t[0:600],inv_yhat[0:600],'b',label="prediction")
#pyplot.legend()
#pyplot.show()