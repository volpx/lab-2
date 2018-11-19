#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:45:48 2018

@author: volpe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from functions import DataXY,general_regression,bode_plot


def import_data():
    df1=pd.read_csv('data/LP_r1.csv',header=None).values
    df2=pd.read_csv('data/LP_r2.csv',header=None).values
    df3=pd.read_csv('data/LP_r3.csv',header=None).values
    df4=pd.read_csv('data/HP.csv',header=None).values
    ds=[df1,df2,df3,df4]
    for df in ds:
        #adjust meas units
        df[:,1]*=1e-3
        df[:,2]*=1e-3
        df[:,0]*=2*np.pi
    return ds

ds=import_data()
r_n=[100e3,10e3,1e3]

#%%
# create H for each filter H[i]=[H_puls,H_mod,H_ph]
H=[np.empty((i.shape[0],3)) for i in ds]
for i in range(4):
    H[i][:,0]=ds[i][:,0]
    H[i][:,1]=ds[i][:,3]
    H[i][:,2]=ds[i][:,4]



#%%

w=2*np.pi*10**np.linspace(0,10,30)
s=1j*w

C=10e-9
R=1e3

ZC=1/s/C
ZR=R*s**0

H=ZC/(ZR+ZC)
w3db=1/C/R

fig,(ax_mod,ax_ph)=bode_plot(w,H,fmt='b,-')
ax_mod.axvline(w3db)
ax_ph.axvline(w3db)
ax_mod.axhline(-3)
ax_ph.axhline(0)
plt.show()