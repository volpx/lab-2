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

data_folder='data4/'
sc_fn='raw_data_from_scope/r{s}_{t}'

# prepare array to save regression parameters
lam= np.empty((5,6,3))
dlam=np.empty((5,6,3))

# %%

# discharge
R_DMM1=np.array([20.16,46.17,98.8,149.4,198.3])

for si in range(5):
    s=si+1
    for ti in range(6):
        t=ti+1

        file=data_folder+sc_fn.format(s=s,t=t)
        print('File:',sc_fn.format(s=s,t=t))
        df=pd.read_csv(file+'.csv',header=0,skiprows=[1])

        x=df['x-axis'].values
        y=df['2'].values

        #find t=0 index
        i_min=0
        while x[i_min] < 0 or np.isnan(x[i_min]):
            i_min+=1
        i_max=i_min
        #find i_max corresponding a 2tau
        while y[i_max] > y[i_min]/np.e**2 :
            i_max+=1

        # restrict to usable data
        x=x[i_min:i_max]
        y=y[i_min:i_max]

        #find deltas
        v_div,t_div = find_divisions(file+'.txt')
        dy=v_div*8*0.03
        dx=t_div*10*8e-4

        #log version
        y_log=np.log(y)
        dy_log=dy/y

#        fig=plt.figure()
#        ax=fig.add_subplot(1,1,1)
#        ax.errorbar(x[::100],y_log[::100],xerr=dx,yerr=dy_log[::100])

        # create array functions
        F=np.vstack( [ np.ones(x.size), x, 1/y] ).T
        (lam[si,ti],
         dlam[si,ti],_,
         _,
         _,_,_) = general_regression( F=F, y=y_log, dy=dy_log )

# %%
# Second stage
# TODO: to consider Rosc
lam_mean=np.mean(lam,axis=1)
dlam_mean=np.std(lam,axis=1) / np.sqrt(6)

x=R_DMM1
y=-lam_mean[:,1]
dy=dlam_mean[:,1]

A = linear_regression_AB(x=x,y=y,w=1/dy**2)

fig_tau,(_,_)=fit_plot(x=x,y=y,model=A[0] + A[1]*x,
                        yerr=dy,x_label='R [Ohm]',y_label='1/tau [s^-1]')

L=1/A[1]
Rl=A[0]/A[1]-50

# %%
# filters RCL
v_div_in=[.5,.5]

i=1

file=data_folder+'RCL{i}.csv'.format(i=i+1)
df=pd.read_csv(file).values

# order values for better plot
ind=np.argsort(df[:,0])
df=df[ind[6:],:]
# adjust measurement units
df[:,1]*=1e-3
df[:,2]*=1e-3

# extract data
w=df[:,0]*2*np.pi
dv_out=8*0.03*df[:,1]      /np.sqrt(16)
dv_in= 8*0.03*v_div_in[i]  /np.sqrt(16)
dt=10*8e-4*df[:,2]         /np.sqrt(16)
H_mod=df[:,4]/df[:,3]
H_ph=df[:,5]/180*np.pi

dH_mod=np.sqrt( (1/df[:,3])**2 * dv_out**2 + (df[:,4]/df[:,3]**2)**2 * dv_in**2 )
dH_ph=np.sqrt(2)*dt*w

# plot experimental data
fig_bode,(_,_) = bode_plot(x=w, H=np.vstack([H_mod,H_ph]).T, #dB=False,
                              Herr=[dH_mod,dH_ph],err=True,title='RCL filter')

# calculate model
Rl=0.5
#TODO: the values are to be checked throughly
#R_DMM2=[466,9924.5]
R_DMM2=[466,9924.5]
R=R_DMM2[i]
L=12.6e-3
Cl=2.27e-11
C=3.16e-8
Cosc=108e-12
Rosc=1.0945e6

z1= par( Rosc, par( 1/(1j*w*(Cosc+C+Cl)), Rl+1j*w*L ) )
H0=z1/(z1+R)

fig_bode,(_,_) = bode_plot(x=w,H=H0,ext_fig=fig_bode,fmt='r,-')


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

print('DATA {i}: R:'.format(i=i),R_DMM2[i],'C:',C)
print('Chi2red_mod:',chi2red_mod,'@ dof:',H_mod.size)
print('Chi2red_ph:',chi2red_ph,'@ dof:',H_mod.size)





# %%
#
#L=12.6e-3
#Cl=2.27e-11
#C=3.16e-8
#
#df=pd.read_csv('data/RCL2.csv',header=1).values
#
#fig,(ax_top,ax_bot)=bode_plot(x=df[:,0]*2*np.pi,
#          H = np.vstack([df[:,4]/df[:,3],df[:,5]]).T,
#          fmt='b.')
#
#w=10**(np.linspace(2,6,100))
#zc=1/1j/w/C
#zl=0.5+1j*w*L
#H=par(zl,zc)/(par(zl,zc)+R_DMM2[1])
#bode_plot(x=w,H=H,fmt='b.')

