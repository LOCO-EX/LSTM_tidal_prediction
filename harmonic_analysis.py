#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to perform the harmonic analysis of the sea level

This script reads the sea level (corrected) raw data to perform harmonic analysis
using utide

It saves the tidal reconstruction, the residual, and the original signal



Must intall utide
conda install utide -c conda-forge


@author: mduranmatute


Created on Sat Nov 27 18:00:13 2021
@author: mduranmatute
"""



import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
#%% Function to convert MATLAB's datenum to Python's datetime
import datetime as dt


def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return dt.datetime.fromordinal(int(datenum)) \
           + dt.timedelta(days=days) \
           - dt.timedelta(days=366)


# %% Load Data

SL = pd.read_csv('../sea_level_ML/data/raw_data/SL_DH_data.csv')

time  = mdates.date2num(list(map(datenum_to_datetime,SL.datenum)))
level = np.array(SL.CorrectedSeaLevel[:])-694.6


# %% Perform harmonic analysis

#import netCDF4
import utide


coef = utide.solve(time, level,
                   lat=52, 
                   method='ols',
                   conf_int='MC')


# %% Perform reconstruction
 
tide = utide.reconstruct(time, coef)

#%%
d = {'time': time[::6], 'level': level[::6], 'tide': tide.h[::6]}
df = pd.DataFrame(data=d)


df.to_csv('data/level_tide.csv')
# %% Plot
#t = obs.index.values  # dtype is '<M8[ns]' (numpy datetime64)
# It is more efficient to supply the time directly as matplotlib
# datenum floats:
t = tide.t_mpl

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharey=True, sharex=True)

ax0.plot(t[0:2000], level[0:2000], label=u'Observations', color='C0')
ax0.plot(t[0:2000], tide.h[0:2000], label=u'Tide Fit', color='C1')
ax2.plot(t[0:2000], level[0:2000] - tide.h[0:2000], label=u'Residual', color='C2')
ax2.xaxis_date()
fig.legend(ncol=3, loc='upper center')
fig.autofmt_xdate()