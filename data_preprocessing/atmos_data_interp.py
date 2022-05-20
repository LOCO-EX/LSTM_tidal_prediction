#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to interpolate atmospheric data from UERRA: increase temporal resolution, and save into speed, cosine of angle and sine of angle, pressure and pressure gradient


Created on Mon Apr 11 15:01:26 2022

@author: Matias Duran-Matute (m.duran.matute@tue.nl)

"""

#%%


import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import datetime

#%% Load date


W = pd.read_csv('../data/uerra_b_1h.csv')



speed = np.array(W['wind_speed'])
angle = np.array(W['wind_direction'])
press = np.array(W['pressure'])
px = np.array(W['pressure'])
py = np.array(W['pressure'])

W.time = pd.to_datetime(W.time, format='%Y-%m-%d %H:%M:%S')
t = np.array((W.time - W.time.dt.floor('D')[0]).dt.total_seconds())

#%% Useful definitions
dt = 600 #dt for the new time series in seconds
gdr = np.pi/180 # useful to transform from degrees to radians

#%%

# Define interpolating functions
# For speed and pressure linear interpolation is used
# For the angle, the cosine of the angle is interpolated using the nearest neighbour.
# Then, the angle is recomputed and used to compute the sine

fs = interp1d(t,speed)
fp = interp1d(t,press)
fd = interp1d(t,np.cos(angle),kind='nearest')
fpx = fp = interp1d(t,px)
fpy = fp = interp1d(t,px)

# Perform the interpolation
t_i = np.arange(t[0],t[-1:],dt) # New time vector

speed_i = fs(t_i)
press_i = fp(t_i)
px_i    = fpx(t_i)
py_i    = fpy(t_i)


c_angle_i = fd(t_i)
angle_i = np.arccos(c_angle_i)
s_angle_i = np.sin(angle_i)

#%%

time = pd.to_datetime(t_i + W.time[0].timestamp()-3600, unit='s') 


# %%

d = {'time': time, 'wind_speed': speed_i, 'sine_wind_angle': s_angle_i, 'cosine_wind_angle': c_angle_i, 'pressure': press_i, 'pressure_x_g': px_i, 'pressure_y_g': py_i}
df = pd.DataFrame(data=d)

df.to_csv('../data/uerra_b_10min.csv')