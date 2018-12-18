#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:51:58 2018

@author: volpe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% FBR
df1=pd.read_csv('data6/FBR.csv').values

Rl=df1[:,0]

fig1=plt.figure()
fig1.suptitle('In/Out characterisic')
ax1=fig1.add_subplot(111)


ax1.plot(Rl,df1[:,1]*np.sqrt(2),'b.',label='Vin max')
ax1.plot(Rl,df1[:,2],'r.',label='Vout RMS')
ax1.set_xscale('log')
ax1.set_xlabel('R load [Ω]')
ax1.set_ylabel('V [V]')
ax1.grid()
ax1.legend()


fig2=plt.figure()
fig2.suptitle('Ripple out')
ax2=fig2.add_subplot(111)

ax2.plot(Rl,df1[:,3],'b.')
ax2.set_xscale('log')
ax2.set_xlabel('R load [Ω]')
ax2.set_ylabel('Vpp [V]')
ax2.grid()

#%% FBRD
