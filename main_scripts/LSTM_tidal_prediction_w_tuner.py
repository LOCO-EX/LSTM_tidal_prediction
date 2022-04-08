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
    
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-3, sampling="log")
        #learning_rate = 0.001
        Adam(lr=learning_rate)
    
        #model.compile(hp.Choice("loss", ["mse", "mae"]), optimizer="adam", metrics=["mse"]) #mean absolute error "mse" "mae"
        model.compile(loss="mse", optimizer="adam", metrics=["mse"]) #mean absolute error "mse" "mae"
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [32, 48, 72, 96]),
            #batch_size = 96
            **kwargs,
        )

# %% Load data
# Load sea level data

p_name = "test_SS_nt2_2y_nin192"

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
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
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
tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective="val_mse",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="./tuner",
    project_name=p_name,
)
tuner.search(train_X, train_y, epochs=120, validation_data=(test_X, test_y))
#tuner.search(train_X, train_y, epochs=2, validation_data=(test_X, test_y), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=10)])


tuner.search_space_summary()

#%% Determine optimal hyperparameters
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters()[0]

#%% Find optimal epoch

model = tuner.hypermodel.build(best_hps)
history = model.fit(train_X, train_y, epochs=120, validation_data=(test_X, test_y))

val_mse_per_epoch = history.history['val_mse']
best_epoch = val_mse_per_epoch.index(min(val_mse_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


# Retrain the model
history_b = model.fit(train_X, train_y, epochs=best_epoch, validation_data=(test_X, test_y))

# plot history
pyplot.plot(history_b.history['loss'], label='train')
pyplot.plot(history_b.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig("./models/loss.png", dpi=150)
pyplot.close()

# %% Save model 

#best_model = tuner.get_best_models(num_models=1)[0]
model.save('./models/')


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
print('Test RMSE: %.3f' % rmse)
print('Test std: %.3f' % inv_y.std())

tuner.search_space_summary()

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('lr')}.
""")

# %% Comparison plots

t = L['time']-L['time'][idi]

pyplot.plot(inv_y, inv_yhat,'o')
pyplot.xlabel("data")
pyplot.ylabel("prediction")
pyplot.grid()
pyplot.axis([500,900,500,900])
pyplot.axis("equal")
pyplot.savefig('./models/comp1.png', dpi=150)
pyplot.close()

pyplot.plot(t[0:inv_y.size],inv_y,'r',label="data")
pyplot.plot(t[0:inv_y.size],inv_yhat,'b:',label="prediction")
pyplot.legend()
pyplot.savefig('./models/comp2.png', dpi=150)
pyplot.close()

pyplot.plot(t[0:600],inv_y[0:600],'r',label="data")
pyplot.plot(t[0:600],inv_yhat[0:600],'b:',label="prediction")
pyplot.legend()
pyplot.savefig('./models/comp3.png', dpi=150)
pyplot.close()

# %% Comparison ffts
freq = np.fft.fftfreq(inv_y.size, d=t[idi+nt])[0:int(inv_y.size/4)]

fft_y = np.abs(np.fft.fft(inv_y))[0:int(inv_y.size/4)]
#fft_y = fft_y[0:int(inv_y.size/2)]
fft_yhat = np.abs(np.fft.fft(inv_yhat))[0:int(inv_yhat.size/4)]

pyplot.plot(freq,fft_yhat)
pyplot.plot(freq,fft_y)
pyplot.yscale('log')
pyplot.savefig('./models/spectrum.png', dpi=150)
pyplot.close()
