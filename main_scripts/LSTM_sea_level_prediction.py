#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script uses an LSTM network in keras to predict the tides (sea level) as a function of astronomical motions

It is based on an example given in
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

For this script to work it is necessary to already have the time series of the relative postion of the Moon and the Sun. 
This time series can be obtained by running the script



@author: Matias Duran-Matute (m.duran.matute@tue.nl)
"""
#%%
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import pandas as pd
import numpy as np
import datetime
from scipy.stats import pearsonr
from pickle import dump
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

p_name = "SL_4"


in_folder = ('../data/')

# Load predictors 
predictors = {"files": ['uerra_10min_dim_a.csv','uerra_10min_dim_b.csv','uerra_10min_dim_c.csv','uerra_10min_dim_d.csv'],
              "keys": ['wind_speed_squared', 'uU', 'vU', 'pressure'],
              }


#Load sea level data
L = pd.read_csv(in_folder+'DenHeld_HA.csv')


#Load atmospheric data
d_in = {}

T = pd.read_csv(in_folder+predictors["files"][0],usecols=['time'])
T['time']=pd.to_datetime(T['time'])
for i in np.arange(0,np.size(predictors["files"])):
    print(predictors["files"][i])
    globals()['D%s' % i] = pd.read_csv(in_folder+predictors["files"][i],usecols=predictors["keys"])
    globals()['D%s' % i] = globals()['D%s' % i].add_suffix('_'+str(i))
    T = pd.concat([T,globals()['D%s' % i]],axis=1,join='inner')

# Load astronomical data
A = pd.read_csv(in_folder+'astronomic_10min.csv')
A['time']=pd.to_datetime(A['time'])

start_date = datetime.datetime(1996,1,1,0,0) #Starting date
end_date = datetime.datetime(2002,1,1,0,0) #End date

#indi = T.loc[start_date]
indi = np.where(T.time > start_date)[0][0]
indf = np.where(T.time <= end_date)[0][-1]
nt = 6 # This can be used to reduce temporal resolution (see following lines)

L = L[indi:indf:nt]
T = T[indi:indf:nt]


indi = np.where(A.time >= start_date)[0][0]
indf = np.where(A.time < end_date)[0][-1]

A = A[indi:indf:nt]

#A['distance_moon_au']=A['distance_moon_au']**(-3)
#A['distance_sun_au']=A['distance_sun_au']**(-3)

A_keys = ['altitude_moon_deg','azimuth_moon_cos','azimuth_moon_sin', 'distance_moon_au',
          'altitude_sun_deg','azimuth_sun_cos', 'azimuth_sun_sin', 'distance_sun_au']

A = A[A_keys]          

dataset = pd.concat([T,A],axis=1,join='inner')
dataset = pd.concat([dataset,L['level']],axis=1,join='inner')
# %%

values = dataset.values[:,1:]

nsamples=values.shape[0] #=14107
n_train_periods = int(nsamples*0.8) #percentage for training
n_test_periods  = int(nsamples*0.2) #percentage for testing

# %%
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = StandardScaler()
sc_fit = scaler.fit(values[:n_train_periods,:])
scaled = scaler.transform(values)

# frame as supervised learning
n_steps_in = 48  #specify the number of the previous time steps to use for the prediction = 1 in this case
n_steps_out = 1 #specify the number of time steps to predict = 1 in this case because we are predicting only 1 time step
n_features = dataset.shape[1]-2 #number of features (variables) used to predict

# frame as supervised learning
reframed = series_to_supervised(scaled, n_steps_in, n_steps_out, n_features)
reframed.shape

# save scaler
dump(scaler, open('../models/scaler.pkl', 'wb'))

# %%
# split into train and test sets

values = reframed.values

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
model.add(LSTM(72,activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]))) #=(n_steps_in,n_features)
#model.add(LSTM(48, activation='relu', return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]))) #=(n_steps_in,n_features)
#model.add(LSTM(12, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))

Adam(lr=0.001)

model.compile(loss='mse', optimizer='adam') #mean absolute error "mse" "mae"

# fit network
history = model.fit(train_X, train_y, epochs=71, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig("../models/loss.png", dpi=150)
pyplot.close()
# %% Save model 

model.save('../models/')
#%% Find optimal epoch

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig("../models/loss.png", dpi=150)
pyplot.close()

# %% Make a prediction
yhat = model.predict(test_X)
test_X0 = test_X.reshape((test_X.shape[0], n_steps_in*n_features))
# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = concatenate((test_X0,yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat[:,-(n_features+1):])
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y0 = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_X[:, -4:]), axis=1)
inv_y = concatenate((test_X0,test_y0), axis=1)
inv_y = scaler.inverse_transform(inv_y[:,-(n_features+1):])
inv_y = inv_y[:,-1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
corr, _ = pearsonr(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
print('Test corr: %.3f' % corr)
print('Test std: %.3f' % inv_y.std())



# %% Comparison plots

t = T['time']-T['time'][indi]

pyplot.plot(inv_y, inv_yhat,'o')
pyplot.plot([-250, 200],[-250, 200],'r')
pyplot.xlabel("data")
pyplot.ylabel("prediction")
pyplot.grid()
pyplot.axis([500,900,500,900])
pyplot.axis("equal")
pyplot.savefig('../models/comp1.png', dpi=150)
pyplot.close()

pyplot.plot(t[0:inv_y.size],inv_y,'r',label="data")
pyplot.plot(t[0:inv_y.size],inv_yhat,'b:',label="prediction")
pyplot.legend()
pyplot.savefig('../models/comp2.png', dpi=150)
pyplot.close()

pyplot.plot(t[0:600],inv_y[0:600],'r',label="data")
pyplot.plot(t[0:600],inv_yhat[0:600],'b:',label="prediction")
pyplot.legend()
pyplot.savefig('../models/comp3.png', dpi=150)
pyplot.close()

# %% Comparison ffts

t2 = t.dt.total_seconds()

freq = np.fft.fftfreq(inv_y.size, d=t2[indi+nt])[0:int(inv_y.size/4)]

fft_y = np.abs(np.fft.fft(inv_y))[0:int(inv_y.size/4)]
#fft_y = fft_y[0:int(inv_y.size/2)]
fft_yhat = np.abs(np.fft.fft(inv_yhat))[0:int(inv_yhat.size/4)]

pyplot.plot(freq,fft_yhat)
pyplot.plot(freq,fft_y)
#pyplot.yscale('log')
pyplot.savefig('../models/spectrum.png', dpi=150)
pyplot.close()
