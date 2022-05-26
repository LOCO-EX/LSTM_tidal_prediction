#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script uses the skyfield pacakge to generate the data files containining the Moon and Sun positions relative to a geographical location on Earth


General comments when using this skyfield package:
#
#- baricentric: barycentric = earth.at(t) or dws.at(t)
#  position of earth center or dws or any other point in space measured from the Solar System’s center of mass (without any corrections)
#- astrometric = barycentric.observe(moon)
#  when you use astrometric you can observe moon from earth center or dws at time t,
#  it applies the effect of light travel time. 
#  i.e, on Earth we see the Moon where it was about 1.3 seconds ago, the Sun where it was 8 minutes ago.
#  Recently, it has been confirmed that "gravity" (according to Einstein) travel at the speed of light,
#  so if the sun "disappears" we will still feel gravity as waves arriving during 8 minutes, read:
#  https://www.forbes.com/sites/startswithabang/2020/12/18/ask-ethan-why-doesnt-gravity-happen-instantly/?sh=343391f17fd2
#- apparent = astrometric.apparent()
#  correct aberration of light produced by the observer’s own motion through space, and the gravitational deflection
#  light that passes close to masses like the Sun and Jupiter — and, for an observer on the Earth’s surface, for deflection produced by the Earth’s own gravity.
#  These corrections seem to be important when looking into the space to find an object, but not
#  for the gravity effect on tides; however, the only way to get azimuth and altitude in this package is using .apparent()
#- altitude,azimuth = apparent.altaz()
#  get altitude and azimuth, can only be called with .apparent()

#Tutorials:
#https://rhodesmill.org/skyfield/positions.html#quick-reference
#
#https://rhodesmill.org/skyfield/time.html#downloading-timescale-files
#https://rhodesmill.org/skyfield/api-time.html#calendar-date
#https://rhodesmill.org/skyfield/example-plots.html
#https://rhodesmill.org/skyfield/positions.html
#https://rhodesmill.org/skyfield/


Created on Sat Feb 26 15:24:26 2022

@author: Matias Duran-Matute (m.duran.matute@tue.nl)

"""
#%%
from skyfield.api import load
from skyfield.api import N, S, W, E, wgs84
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

#%%
# General Definitions
#======================

#Needed input from user:

#input dates (should be in UTC)
ts = load.timescale()
t0=ts.tt(1996,1,1,0,0) #initial date: t0.tt = elapsed time in days since Julian date zero
t1=ts.tt(2016,1,1,0,0) #final date: t1.tt = elapsed time in days since Julian date zero
dt=10 #time step in minutes

#Topocentric coordinates specific to your location on the Earth’s surface------
lat=52.96; lon=4.79 #Examples: Den Helder, The Netherlands)

# =======================
# %%
#Load the JPL ephemeris DE421 (covers 1900-2050)---
#An ephemeris from the JPL provides Sun, Moon, planets and Earth positions.
eph = load('de421.bsp')
earth, moon, sun = eph['earth'], eph['moon'], eph['sun']
dws = earth + wgs84.latlon(lat*N,lon*E) 
#----
#dates in utc also are referenced using time coordinate tt
#TT: terrestrial time, this uses relative time to a certain epoch (Julian date)
#so, this is an uniform or absolute time scale, and better for skyfield computations because what only matters is the time difference.
#When was Julian date zero?
#bc_4714 = -4713
#t = ts.tt(bc_4714, 11, 24, 12)


#Build a time vector from a UTC calendar----
#there are 21 leap seconds lag when using skyfield utc

times = ts.tt(1996, 1, 1, 0, range(0,int((t1-t0)*24*60),dt))

#check the diff of time is still 1h
dtt=np.diff(times.tt)*24 
#print(dtt.min(),dtt.max(),times.shape)
#check the first 2 and the last 2 times of the vector built with skynet:
#there are 21 leap seconds from 1980-2015
#but we still have the same time difference (elapsed time) when using for example numpy datetime64 (see below cell)
print(times[[0,1]].tt_strftime('%Y-%m-%d %H:%M:%S'))
print(times[[-2,-1]].tt_strftime('%Y-%m-%d %H:%M:%S'))



#%%
# 1) Moon
#MOON----
astro_moon = dws.at(times).observe(moon)
#- altitude or elevation (deg,min,sec): -90:90deg
#- azimuth (deg,min,sec): 0:360deg
#- distance (au or km): 1au (astronomical unit) = 1.496e8km
alt_moon, az_moon, dist_moon = astro_moon.apparent().altaz()

#convert above values to arrays---
#altitude_moon_rad=alt_moon.radians
altitude_moon_deg=alt_moon.degrees
#azimuth_moon_rad=alt_moon.radians
azimuth_moon_deg=az_moon.degrees
distance_moon_au = dist_moon.au
#%%
# 2) Sun

astro_sun = dws.at(times).observe(sun)
#- altitude or elevation (deg,min,sec): -90:90deg
#- azimuth (deg,min,sec): 0:360deg
#- distance (au or km)
alt_sun, az_sun, dist_sun = astro_sun.apparent().altaz()

#convert above values to arrays---
#altitude_sun_rad=alt_sun.radians
altitude_sun_deg=alt_sun.degrees
#azimuth_sun_rad=alt_sun.radians
azimuth_sun_deg=az_sun.degrees
distance_sun_au = dist_sun.au #AU units
#distance_sun_km = dist_sun.km #km units
#distance_sun_km2 = dist_sun2.km #km units
#a=distance_sun_km-distance_sun_km2 #almost the same distnace when using .radec


# Save data
#%%
#t_num = mdates.date2num(times.utc_datetime())

d = {'time':times.tt_strftime('%Y-%m-%d %H:%M:%S'), 'altitude_moon_deg': altitude_moon_deg, 'azimuth_moon_deg': azimuth_moon_deg, 'distance_moon_au':distance_moon_au,'altitude_sun_deg': altitude_sun_deg, 'azimuth_sun_deg': azimuth_sun_deg, 'distance_sun_au':distance_sun_au}
df = pd.DataFrame(data=d)

df.to_csv('../data/astronomic_10min.csv')
