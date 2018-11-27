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
from functions import DataXY,general_regression,bode_plot,par


# discharge
R_DMM1=[20.16,46.17,98.8,149.4,198.3]

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

# %%
R_DMM2=[466,9924.5]
L=12.6e-3
Cl=2.27e-11
C=3.16e-8

df=pd.read_csv('data/RCL2.csv',header=1).values

fig,(ax_top,ax_bot)=bode_plot(x=df[:,0]*2*np.pi,
          H = np.vstack([df[:,4]/df[:,3],df[:,5]]).T,
          fmt='b.')

w=10**(np.linspace(2,6,100))
zc=1/1j/w/C
zl=0.5+1j*w*L
H=par(zl,zc)/(par(zl,zc)+R_DMM2[1])
bode_plot(x=w,H=H,fmt='b.')

