#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to extract atmospheric data from hindcast products in netcdf files


Created on Tue Apr 12 08:57:02 2022

@author: Matias Duran-Matute (m.duran.matute@tue.nl)
"""

#%%

import netCDF4 as nc
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#%%
input_folder = "/Users/matias/Atmosphere/"

fr = "UERRA"

years = range(1996,2016)

#File a: Texel inlet
#nx = 18 # lat = 52.8
#ny = 10 # lon = 4.92
#File b: Towards the channel
#nx = 0
#ny = 0
#File c: northwest corner
nx = 0
ny = -1

v = np.empty(0,dtype=float)
a = np.empty(0,dtype=float)
p = np.empty(0,dtype=float)
px = np.empty(0,dtype=float)
py = np.empty(0,dtype=float)
t  = np.empty(0,dtype=object)

#%%
for i in years:
	fn = input_folder + fr + "." + str(i) + ".nc4"
	print(fn)
	ds = nc.Dataset(fn)
	
	time_var = ds.variables['time']
	dtime = nc.num2date(time_var[1:-24], time_var.units)

	
	press = np.array(ds['slp'][1:-24,nx,ny])
	#pgrad = np.gradient(np.squeeze(np.array(ds['slp'][1:-24,(nx-2):(nx+3),(ny-2):(ny+3)])),axis = [1, 2])
	u10     = np.array(ds['u10'][1:-24,nx,ny])
	v10     = np.array(ds['v10'][1:-24,nx,ny])
	
	t = np.append(t, dtime.data)
	p = np.append(p, press)
	v = np.append(v, np.sqrt(u10**2 + v10**2))
	a = np.append(a, np.arctan2(u10,v10))
	#px = np.append(px, pgrad[0][:,2,2])
	#py = np.append(py, pgrad[1][:,2,2])

#%%

#time = pd.DataFrame(dtime.data)

#d = {'time': t, 'wind_speed': v, 'wind_direction': a, 'pressure': p, 'pressure_x_g': px, 'pressure_y_g': py}
d = {'time': t, 'wind_speed': v, 'wind_direction': a, 'pressure': p}

df = pd.DataFrame(data=d)

df.to_csv('../data/uerra_1h_c.csv')