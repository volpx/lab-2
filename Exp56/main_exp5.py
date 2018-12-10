#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:51:58 2018

@author: volpe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% DIODE CHARACTERISTIC
plt.figure()
for i in range(1):
    df=pd.read_csv('data5/D{i}.csv'.format(i=i+1),header=0).values
    df[:,1]*=1e-3
    plt.plot(df[:,2],df[:,1],'b.')
plt.yscale('log')
plt.grid()

# %% ZENER CHARACTERISTIC
plt.figure()
df=pd.read_csv('data5/Z.csv'.format(i=i+1),header=0).values
df[:,1]*=1e-3
plt.plot(df[:,2],df[:,1],'b.-')
#plt.yscale('log')
plt.grid()

#dynamic resistance
rd=(df[1:,2]-df[:-1,2])/(df[1:,1]-df[:-1,1])

plt.figure()
plt.plot(rd,df[1:,1],'b.')
plt.yscale('log')
plt.xscale('log')
plt.grid()