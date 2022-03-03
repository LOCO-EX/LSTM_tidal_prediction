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
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import pandas as pd
import numpy as np
import datetime


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

#%% Define model with tuner

class MyHyperModel(kt.HyperModel):

    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int("units", min_value=24, max_value=64, step=8), input_shape=(train_X.shape[1], train_X.shape[2]))) #=(n_steps_in,n_features)
        model.add(Dense(1))
    
        #learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-3, sampling="log")
        learning_rate = 0.001
        Adam(lr=learning_rate)
    
        model.compile(hp.Choice("loss", ["mse", "mae"]), optimizer='adam', metrics=["accuracy"]) #mean absolute error "mse" "mae"
    
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [32, 48]),
            **kwargs,
        )

# %% Load data
# Load sea level data

#L = pd.read_csv('data/level_DH_10min.csv')

L = pd.read_csv('data/SL_DH_decomposed.csv')
L["level"] = L["tide"][:]

# Load astronomic data
A = pd.read_csv('data/astronomic_10min.csv')
A = A[:-1]


# %% 
ti = datetime.datetime(1996,1,1,0,0) #Starting date
tf = datetime.datetime(1998,7,1,1,0) #End date

ti_d = ( ti - datetime.datetime(1970,1,1)).total_seconds()/86400.
tf_d = ( tf - datetime.datetime(1970,1,1)).total_seconds()/86400.

idi = (np.abs(L['time']-ti_d)).argmin()
idf = (np.abs(L['time']-tf_d)).argmin()


nt = 2 # This can be used to reduce temporal resolution (see following lines)

L = L[idi:idf:nt]
A = A[idi:idf:nt]
#%% Moon and sun azimuth into sine and cosine

gdr = np.pi/180 # useful to transform from degrees to radians

ma_cos = np.cos(A['azimuth_moon_deg']*gdr)
ma_sin = np.sin(A['azimuth_moon_deg']*gdr)
sa_cos = np.cos(A['azimuth_sun_deg']*gdr)
sa_sin = np.sin(A['azimuth_sun_deg']*gdr)

# %% Arrange data
# FULL INPUT (three variables for each Moon and Sun position)
tmp = np.stack((A['altitude_moon_deg'], A['distance_moon_au'], ma_cos, ma_sin, A['altitude_sun_deg'], A['distance_sun_au'], sa_cos, sa_sin,L['level'][0:ma_cos.shape[0]]))
d = {'altitude_moon_deg': tmp[0,:], 'distance_moon_au': tmp[1,:]**(-3), 'azimuth_moon_cos': tmp[2,:], 'azimuth_moon_sin': tmp[3,:], 'altitude_sun_deg': tmp[4,:], 'distance_sun_au': tmp[5,:]**(-3), 'azimuth_sun_cos': tmp[6,:], 'azimuth_sun_sin': tmp[7,:], 'level': tmp[8,:]}

# ONLY MOON
#tmp = np.stack((A['altitude_moon_deg'], A['distance_moon_au'], ma_cos, ma_sin,tide['h']))
#in_pein_periods:(n_test_periods+n_train_periods)riods:(n_test_periods+n_train_periods)d = {'altitude_moon_deg': tmp[0,:], 'distance_moon_au': tmp[1,:], 'azimuth_moon_cos': tmp[2,:], 'azimuth_moon_sin': tmp[3,:], 'level': #tmp[4,:]}

dataset = pd.DataFrame(data=d)
values = dataset.values

nsamples=values.shape[0] #=14107
n_train_periods = int(nsamples*0.7) #percentage for training
n_test_periods  = int(nsamples*0.3) #percentage for testing

# %%
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
sc_fit = scaler.fit(values[:n_train_periods,:])
scaled = scaler.transform(values)

# frame as supervised learning
n_steps_in = 192  #specify the number of the previous time steps to use for the prediction = 1 in this case
n_steps_out = 1 #specify the number of time steps to predict = 1 in this case because we are predicting only 1 time step
n_features = 8 #number of features (variables) used to predict

# frame as supervised learning
reframed = series_to_supervised(scaled, n_steps_in, n_steps_out, n_features)
reframed.shape

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
tuner = kt.RandomSearch(
    MyHyperModel(),
    objective="mse",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="./tuner",
    project_name="LSTM_SL",
)
tuner.search(train_X, train_y, epochs=2, validation_data=(test_X, test_y))
#tuner.search(train_X, train_y, epochs=2, validation_data=(test_X, test_y), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=10)])


tuner.search_space_summary()
