#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates test (artificial) tidal signal based on a real tidal signal using harmonic analysis and the residual

This script uses the python version of the utide package:
    Info on original packages: http://www.po.gso.uri.edu/~codiga/utide/utide.htm
    Python implementation: https://github.com/wesleybowman/UTide
    

"""
#from math import sqrt
#from numpy import concatenate
#from pandas import read_csv
#from pandas import DataFrame
#from pandas import concat
import pandas as pd
#import numpy as np
import utide


# %% Load data
# Load sea level data
L = pd.read_csv('data/level_DH_10min.csv')




# %%

#If you want to substitute L['tide'] by a specific reconstruction
coef = utide.solve(L['time'], L['level'],
                   lat=52.96, 
                   method='ols',
                   conf_int='MC')


# const =  ['M2', 'S2', 'M4', 'O1', 'SA', 'N2', 'MU2', 'K1', 'MS4', 'L2', 'M6',
#          '2MS6', 'K2', 'NU2', 'MN4', 'Q1', 'P1', '2MN6', 'LDA2', 'M8', 'H1',
#          'SSA', 'MK4', 'EPS2', 'MSN2', '2MK6', 'S1', '2MK5', 'MM', 'T2',
#          '2SM6', '2N2', 'MKS2', 'MO3', 'MSK6', 'MSF', 'RHO1', 'SN4', '2Q1',
#          'H2', 'M3', 'S4', 'SO1', 'OO1', 'TAU1', 'MK3', 'SK4', 'GAM2',
#          'PSI1', 'J1', 'ETA2', 'SIG1', 'ALP1', 'PI1', 'SO3', 'BET1', '3MK7',
#          'MSM', 'CHI1', 'MF', 'R2', 'NO1', 'PHI1', 'OQ2', 'SK3', 'THE1',
#          'UPS1', '2SK5']

# Only moon
#const = ['M2', 'N2', 'M4', 'O1', 'K2', 'MU2', 'K1', 'MN', 'L2']


# Most important
#const = ['M2', 'S2', 'M4', 'O1', 'SA', 'N2', 'MU2', 'K1', 'MS4', 'L2', 'M6', '2MS6', 'K2']

#tide = utide.reconstruct(L['time'], coef, constit=const)

tide = utide.reconstruct(L['time'],coef)


res = L['level'] - tide.h

d = {'time': L['time'], 'tide': tide , 'residual': res}
df = pd.DataFrame(data=d)

#%% Compute a smooth version of the residual

df['res_smooth'] = df['residual'].rolling(window = 240, center=True, min_periods=1).mean()
 
#%% Save signal to file
df.to_csv('data/SL_DH_decomposed.csv')

