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
from functions import *

data_folder='data3/'
v_div_in=[.26,.5,.5,.5]
files=['LP1.csv','LP2.csv','LP3.csv','HP1.csv']
R_dmm=[9938,997.7,99.56,]
C=32.892e-9

# %%

i=0
# read in data
df=pd.read_csv(data_folder+files[i]).values

# find uncertainties on data points
df[:,1]*=1e-3
dv_out=8*0.03*df[:,1]     /np.sqrt(16)
dv_in=8*0.03*v_div_in[i]  /np.sqrt(16)
df[:,2]*=1e-3
dt=10*8e-4*df[:,2]        /np.sqrt(16)

# extract transfer function with each uncertainties
w=df[:,0]*2*np.pi
dw=0
H_mod=df[:,4]/df[:,3]
H_ph=df[:,5]/180*np.pi
dH_mod=np.sqrt( (1/df[:,3])**2 * dv_out**2 + (df[:,4]/df[:,3]**2)**2 * dv_in**2 )
dH_ph=np.sqrt(2)*dt*w

# plot the experimental data
fig,(_,_) = bode_plot(x = w,
                                H = np.vstack([H_mod,H_ph]).T,
                                xerr=dw, Herr=[dH_mod,dH_ph],
                                err=True,dB=True,fmt='b.',title='RC filter')

# calculate the model
R=R_dmm[i]

ZC=1/1j/w/C
ZR=R

H0 = ZC / ( ZR + ZC )

# and add to a comp plot
fig,(_,_) = bode_plot(x = w,
                                H = H0,
                                dB=True,fmt='r,-',
                                ext_fig=fig)

# residuals
H_mod_res = H_mod - np.abs(H0)
H_ph_res  = H_ph  - np.angle(H0)

fig_res_bode = plt.figure()
fig_res_bode.suptitle('Bode plot residuals')

ax_top=fig_res_bode.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
ax_bot=fig_res_bode.add_axes([0.1, 0.1, 0.8, 0.4])

ax_top.set_xscale('log')
ax_bot.set_xscale('log')

ax_top.set_ylabel('|H| res []')
ax_bot.set_ylabel('φ res [rad]')
ax_bot.set_xlabel('ω [rad * s^-1]')

ax_top.grid()
ax_bot.grid()

ax_top.errorbar(x=w, y=H_mod_res,
                xerr=0, yerr=dH_mod,
                fmt='b.')
ax_bot.errorbar(x=w, y=H_ph_res,
                xerr=0, yerr=dH_ph,
                fmt='b.')

ax_top.axhline(0,color='r')
ax_bot.axhline(0,color='r')

chi2red_mod=chi2red(H_mod_res,dH_mod)
chi2red_ph=chi2red(H_ph_res,dH_ph)

print('DATA {i}: R:'.format(i=i),R_dmm[i],'C:',C)
print('Chi2red_mod:',chi2red_mod,'@ dof:',H_mod.size)
print('Chi2red_ph:',chi2red_ph,'@ dof:',H_mod.size)

