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

ntau=4
for si in range(5):
#for si in range(1):
    s=si+1
    for ti in range(6):
#    for ti in range(1):
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
        while y[i_max] > y[i_min]/np.e**ntau :
            i_max+=1

        # first very very rough approximation of tau
        tau=(x[i_max]-x[i_min])/ntau

        # discard some data from beginning
        deltat=x[i_max+1]-x[i_max]
        i_min=i_min+int(0.5*tau/deltat)

        # restrict x,y to usable data
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

#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.errorbar(x=x[::25],y=y[::25],yerr=dy,xerr=dx,fmt='b.')
#ax.grid()
#ax.set_ylabel('Vout [V]')
#ax.set_xlabel('Time [s]')
#fig.suptitle('Scarica induttanza')
#fig.savefig('data4/ind_dis.pdf',bbox_inches='tight')

# %%
# Second stage
# TODO: to consider Rosc
lam_mean=np.mean(lam,axis=1)
dlam_mean=np.std(lam,axis=1) / np.sqrt(6)

x=par(R_DMM1,1e6)
y=-lam_mean[:,1] #1/tau
dy=dlam_mean[:,1]

A = linear_regression_AB(x=x,y=y,w=1/dy**2)

fig_tau1,(_,_)=fit_plot(x=x,y=y,model=A[0] + A[1]*x,
                        yerr=dy,x_label='R [Ohm]', y_label='1/tau [s^-1]')

L=1/A[1]
dL=1/A[1]**2 * A[3]
Rl=A[0]/A[1]-50
#max uncertainty
dRl= (A[0]/A[1]**2) * A[3] + (1/A[1]) * A[2]

y_res=y-A[0]-A[1]*x

chi2red_taures=chi2red(y_res,dy=dy,ddof=2)
# %%
# now i say that i dont trust the uncertainties on data because of the chi2 high
dy=dy*np.sqrt(chi2red_taures)

B = linear_regression_AB(x=x,y=y,w=1/dy**2)

fig_tau2,(_,_)=fit_plot(x=x,y=y,model=A[0] + A[1]*x,
                        yerr=dy,x_label='R [Ohm]',y_label='1/τ [s^-1]')

L=1/B[1]
dL=1/B[1]**2 * B[3]
Rl=B[0]/B[1]-50
#max uncertainty
dRl= (B[0]/B[1]**2) * B[3] + (1/B[1]) * B[2]

fig_tau2.suptitle('Regressione lineare τ R')
fig_tau2.savefig('data4/fig_tau2.pdf',bbox_inches='tight')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

#
# plot experimental data
fig_bode,(_,_) = bode_plot(x=w, H=np.vstack([H_mod,H_ph]).T, #dB=False,
                              Herr=[dH_mod,dH_ph],err=True,title='RCL filter R=9924.5Ω')

# %%

# calculate model
#Rl=5.2
Rl=12.41280209327403
#TODO: the values are to be checked throughly
R_DMM2=[466,9924.5]
R=R_DMM2[i]
L=12.68e-3
Cl=0
# C=3.289e-8 - 8.9e-10
C=3.289e-8
Cosc=1.3e-10
Rosc=1.01e6

z1= par( Rosc, par( 1/(1j*w*(Cosc+C+Cl)), Rl+1j*w*L ) )
H0=z1/(z1+R)

fig_bode,(_,_) = bode_plot(x=w,H=H0,ext_fig=fig_bode,fmt='r,-')


fig_bode.savefig('data4/bodefilter2.pdf',bbox_inches='tight')

# %%
# residuals
H_mod_res = H_mod - np.abs(H0)
H_ph_res  = H_ph  - np.angle(H0)

fig_res_bode = plt.figure()
fig_res_bode.suptitle('Bode plot residuals R=9924.5Ω')

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

fig_res_bode.savefig('data4/bodefilter2_res.pdf',bbox_inches='tight')

chi2red_mod=chi2red(H_mod_res,dH_mod)
chi2red_ph=chi2red(H_ph_res,dH_ph)

print('DATA {i}: R:'.format(i=i),R_DMM2[i],'C:',C)
print('Chi2red_mod:',chi2red_mod,'@ dof:',H_mod.size)
print('Chi2red_ph:',chi2red_ph,'@ dof:',H_mod.size)

# %%
# fit the second plot with a parabola
ind=np.arange(8,17)
#ind=np.arange(7,18)

#restrict data
w_f=w[ind]
H_mod_f=H_mod[ind]
H_ph_f=H_ph[ind]
dH_mod_f=dH_mod[ind]
dH_ph_f=dH_ph[ind]

w_f_log=np.log10(w_f)
H_mod_f_db=20*np.log10(H_mod_f)
dH_mod_f_db=20/np.log(10)* dH_mod_f/H_mod_f

F=np.vstack([np.ones(w_f_log.size),w_f_log,w_f_log**2]).T
(lama,dlama,_,_,_,_,_) = general_regression(F=F,y=H_mod_f_db,dy=dH_mod_f_db)

# take the max
w_max = 10**(-lama[1]/2/lama[2])
#dw_max =
H_mod_max = 10**((lama[0]+lama[1]*(-lama[1]/2/lama[2])+lama[2]*(-lama[1]/2/lama[2])**2)/20)
#dH_mod_max =

a=lama[2]
b=lama[1]
c=lama[0]-20*np.log10(H_mod_max/np.sqrt(2))

w1_3db=10**((-b-np.sqrt(b**2-4*a*c))/2/a)
w2_3db=10**((-b+np.sqrt(b**2-4*a*c))/2/a)

Q=w_max/(w1_3db-w2_3db)

Rl1=np.sqrt( 1/((H_mod_max*Rosc*R/(Rosc-H_mod_max*(Rosc+R)))**2-1/(w_max**2*C**2)) )*L/C


# recalculate more fit
w1=10**np.linspace(np.log10(47400),np.log10(52000),100)
z1= par( Rosc, par( 1/(1j*w1*(Cosc+C+Cl)), Rl+1j*w1*L ) )
H01=z1/(z1+R)

#plotit
figp=plt.figure()
figp.suptitle('Parabola fit')
axp=figp.add_subplot(111)
#axp.set_xscale('log')
axp.set_ylabel('|H|dB')
axp.set_xlabel('ω [rad * s^-1]')
axp.grid()
axp.errorbar(x=w_f,y=H_mod_f_db,yerr=dH_mod_f_db,xerr=0,fmt='b.')
axp.plot(w1,20*np.log10(np.abs(H01)),'r,-' )
axp.plot(w1,lama[0]+lama[1]*np.log10(w1)+lama[2]*np.log10(w1)**2,'g,-')
axp.axvline(w_max,color='black')
#axp.axhline(20*np.log10(H_mod_max),color='fucsia')
axp.axhline(20*np.log10(H_mod_max/np.sqrt(2)),color='orange')
axp.axvline(w1_3db,ymax=.5,color='violet')
axp.axvline(w2_3db,ymax=.5,color='violet')
axp.set_ylim([-6,0])
figp.savefig('data4/bodeconf.pdf',bbox_inches='tight')

# %%
# verify that the RCL filter is like a derivator for f<fc and integrator for f>fc
