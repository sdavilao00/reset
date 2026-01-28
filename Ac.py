# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 15:32:11 2025

@author: sdavilao
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Critical Depth variables
pw = 1000 # kg/m3 # density of water
ps = 1600 # kg/m3c # density of soil
g = 9.81  # m/s2 # force of gravity
yw = g*pw
ys = g*ps
phi = np.deg2rad(41)  # converts phi to radians ## set to 42 right now
# z = np.arange(0,6,0.1)
z = 1.03

# Slope stability variables
m = 1 # m # saturation ration (h/z)
l = 10 # m # length
w = 6.7 # m # width
C0 = 1920 # Pa
j = 0.8

#Define side/hollow slope range
hollow_rad = np.deg2rad(filtered_df['Avg_Slope'])


#Cohesion variables
Crb = C0*2.718281**(-z*j)
Crl = (C0/(j*z))*(1 - 2.718281**(-z*j))

K0 = 1 - np.sin(hollow_rad)

Kp = np.tan((np.deg2rad(45))+(phi/2))**2
Ka = np.tan((np.deg2rad(45))-(phi/2))**2
#%% Area

# Define terms of equation
A = (2*Crl*z + K0*(z**2)*(ys-yw*(m**2))*np.tan(phi))*np.cos(hollow_rad)*(l/w)**0.5
B = (Kp-Ka)*0.5*(z**2)*(ys-yw*(m**2))*(l/w)**(-0.5)
C = (np.sin(hollow_rad)*np.cos(hollow_rad)*z*ys) - Crb - (((np.cos(hollow_rad))**2)*z*(ys-yw*m)*np.tan(phi))

#Find critical area
Ac = ((A + B)/C)**2