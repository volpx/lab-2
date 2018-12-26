#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:39:45 2018

@author: volpe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
df=pd.read_csv('data6/data_from_scope/a.csv',skiprows=2).values
noc=1.01*np.abs(.15+8.3*np.cos(df[:,0]*2*np.pi*50+np.pi/180*75))

fig=plt.figure()
fig.suptitle('Ripple capacitor')
ax=fig.add_subplot(111)
ax.plot(df[:,0],df[:,1])
ax.plot(df[:,0],noc)
ax.set_ylim([0,9])
ax.set_ylabel('Vc [V]')
ax.set_xlabel('time [s]')
fig.savefig('report/cripplefilter.pdf')

#%%
df=pd.read_csv('data6/data_from_scope/a1.csv',skiprows=2).values
df2=pd.read_csv('data6/data_from_scope/a2.csv',skiprows=2).values

fig2=plt.figure()
fig2.suptitle('Ripple comparison')
ax2=fig2.add_subplot(111)
ax2.plot(df[:,0],df[:,2],label='Vc pp')
ax2.plot(df2[:,0],df2[:,2],label='Vout pp')
ax2.set_xlim([-0.02,0.02])
ax2.set_ylabel('Vpp [V]')
ax2.set_xlabel('time [s]')
ax2.legend()
fig2.savefig('report/ripplecomp.pdf')