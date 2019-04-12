#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:51:58 2018

@author: volpe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat
plt.rcParams["figure.figsize"] = (8,4)

from functions import *
from numpy import array as na

#coil radius
r=17.5e-3/2
#wire diameter
wd=300e-6
#windings
N=32

#%% Attached coils frequency sweep from 1kHz to 200kHz
fs=na([1e3,1.5e3,2.1e3,3.1e3,4.5e3,6.6e3,9.7e3,14e3,
      21e3,30e3,44e3,64e3,94e3,134e3,200e3])
#15 frequencies, 5 measurements, 2 real and imaginary part
fit_sweep_IN=np.empty((15,5,2))
fit_sweep_OUT=np.empty((15,5,2))
for i in range(75):
    # import the dataframe
    fname='b{}.csv'.format(i+1)
    df=pd.read_csv('data/osccapt2/'+fname,skiprows=[1],header=0)
    
    # extract the quantities
    t=df['x-axis']
    x_in=df['1']
    x_out=df['2']
    f=fs[i//5]
    w=2*np.pi*f
    dx=0.03*np.max(x_in)
    
    # first tmp fit to find to then subtract the phase
    fit_tmp=fit_sine_poly(t=t, x=x_in, polyord=0, freqs=[f], err=dx)
    
    # extract fit
    V0=fit_tmp[0][0]
    A=fit_tmp[0][1]
    B=fit_tmp[0][2]
    C=np.sqrt(A**2+B**2)
    phi=-np.arctan2(B,A)
    t0=-(phi/w)
    
    # refit
    fit_IN=fit_sine_poly(t=t, x=x_in, polyord=0, freqs=[f], err=dx, t0=t0)
    fit_OUT=fit_sine_poly(t=t, x=x_out, polyord=0, freqs=[f], err=dx, t0=t0)
    
    # save the real and imaginary part
    fit_sweep_IN[i//5,i%5]=fit_IN[0][1:3]*na([1,-1])
    fit_sweep_OUT[i//5,i%5]=fit_OUT[0][1:3]*na([1,-1])
    #dfit_sweep_IN[i//5,i%5]=fit_IN[1]
    #dfit_sweep_OUT[i//5,i%5]=fit_OUT[1]

sweep_IN=np.empty((15,2))
sweep_OUT=np.empty(sweep_IN.shape)
dsweep_IN=np.empty(sweep_IN.shape)
dsweep_OUT=np.empty(sweep_IN.shape)

#Mean fits sweep
sweep_IN=np.mean(fit_sweep_IN,axis=1)
sweep_IN_mod=np.sqrt(sweep_IN[:,0]**2+sweep_IN[:,1]**2)
sweep_IN_ph=np.arctan2(sweep_IN[:,1],sweep_IN[:,0])

sweep_OUT=np.mean(fit_sweep_OUT,axis=1)
sweep_OUT_mod=np.sqrt(sweep_OUT[:,0]**2+sweep_OUT[:,1]**2)
sweep_OUT_ph=np.arctan2(sweep_OUT[:,1],sweep_OUT[:,0])

# take the std as uncertainty
dsweep_IN=np.std(fit_sweep_IN,axis=1)
dsweep_OUT=np.std(fit_sweep_OUT,axis=1)
dsweep_OUT_mod=np.sqrt((sweep_OUT[:,0]/sweep_OUT_mod)**2*dsweep_OUT[:,0]**2+\
                       (sweep_OUT[:,1]/sweep_OUT_mod)**2*dsweep_OUT[:,1]**2)
dsweep_OUT_ph=np.sqrt( (1/(1+(sweep_OUT[:,1]/sweep_OUT[:,0])**2)/sweep_OUT[:,0])**2*dsweep_OUT[:,1]+\
                        (1/(1+(sweep_OUT[:,1]/sweep_OUT[:,0])**2)*sweep_OUT[:,1]/sweep_OUT[:,0]**2)**2*dsweep_OUT[:,0])

#%% Plots

#fig1=plt.figure()
#fig1.suptitle('0 cm, frequency sweep 1÷200kHz')
#ax1=fig1.add_subplot(111)
#ax1.errorbar(x=fs,y=sweep_OUT)
fig1,(ax11,ax12)=bode_plot(x=fs*2*np.pi,H=np.vstack([sweep_OUT_mod,sweep_OUT_ph*180/np.pi]).T,
                      Herr=np.vstack([dsweep_OUT_mod,dsweep_OUT_ph]),y_label='Vout [V]',
                      err=False,xerr=0,
                     title='0 cm, frequency sweep 1÷200kHz, Out')
ax11.set_yscale('log')
ax12.set_ylabel('Phase [deg]')

ax11.grid(b=True,axis='x',which='minor',alpha=0.5)
ax12.grid(b=True,axis='x',which='minor',alpha=0.5)
#
fig2,(ax21,ax22)=bode_plot(x=fs*2*np.pi,H=np.vstack([sweep_IN_mod,sweep_IN_ph*180/np.pi]).T,
                      y_label='VRlim [V]',
                     title='0 cm, frequency sweep 1÷200kHz, on Rlim=47Ω')
ax21.set_yscale('log')
ax22.set_ylabel('Phase [deg]')
ax21.set_yticks(np.exp(np.linspace(np.log(2.21),np.log(2.16),5)))

ax21.grid(b=True,axis='both',which='minor',alpha=0.5)
ax22.grid(b=True,axis='x',which='minor',alpha=0.5)
fig2.savefig('fig_tau2.pdf',bbox_inches='tight')
#%%
Cosc=110e-12
Rosc=1e6
Cout=100e-9
Gdiff=35.65
Rlim=47
Rc=1e4
Gdiff_model=lambda w: Gdiff * par(1/(1j*w*Cosc),Rosc)/ \
            (par(1/(1j*w*Cosc),Rosc)+Rc+1/(1j*w*Cout))

iS=(sweep_IN[:,0]+1j*sweep_IN[:,1])/Rlim
Zeff=(sweep_OUT[:,0]+1j*sweep_OUT[:,1])/Gdiff_model(fs*2*np.pi)/iS

MRS=(sweep_OUT[:,0]+1j*sweep_OUT[:,1])/Gdiff_model(fs*2*np.pi)/iS/(2*np.pi*fs)/1j
MRS_mean=np.mean(MRS)
print('MRS_mean=',MRS_mean)
print('MRS_abs=',np.abs(MRS_mean))
print('MRS_phase=',np.angle(MRS_mean))

fig3,(ax31,ax32)=bode_plot(x=fs*2*np.pi,H=Zeff,
                     y_label='Zeff [Ω]',title='Zeff(ω) comparison with model')
fig3,(_,_)=bode_plot(x=fs*2*np.pi,H=1j*fs*2*np.pi*np.abs(MRS_mean), fmt='r-',
                     ext_fig=fig3)
#fig3,(_,_)=bode_plot(x=fs*2*np.pi,H=1j*fs*2*np.pi*MRS, fmt='g-',
                     #ext_fig=fig3)
ax31.grid(b=True,which='minor',axis='x',alpha=0.5)
ax32.grid(b=True,which='minor',axis='x',alpha=0.5)


#%% Distance and frequency sweep from 2cm to 20cm and from 1kHz to 200kHz 
ds=na([2,2.4,4.3,6.3,9.3,13.6,20])*1e-2
fs1=na([1,4.5,30,200])*1e3
#7 distances,4 frequencies*, 5 measurements, 2 real and imaginary part
fit_d0_IN=np.empty((4,5,2))
fit_d0_OUT=np.empty(fit_d0_IN.shape)
fit_d1_IN=np.empty((4,5,2))
fit_d1_OUT=np.empty(fit_d1_IN.shape)
fit_d2_IN=np.empty((4,5,2))
fit_d2_OUT=np.empty(fit_d2_IN.shape)
fit_d3_IN=np.empty((4,5,2))
fit_d3_OUT=np.empty(fit_d3_IN.shape)
fit_d4_IN=np.empty((4,5,2))
fit_d4_OUT=np.empty(fit_d4_IN.shape)
fit_d5_IN=np.empty((3,5,2))
fit_d5_OUT=np.empty(fit_d5_IN.shape)
fit_d6_IN=np.empty((2,5,2))
fit_d6_OUT=np.empty(fit_d6_IN.shape)

def sine_fits(fname,f):
    # import the dataframe
    df=pd.read_csv('data/osccapt2/'+fname,skiprows=[1],header=0)
    
    # extract the quantities
    t=df['x-axis']
    x_in=df['1']
    x_out=df['2']
    w=2*np.pi*f
    dx=0.03*np.max(x_in)
    
    # first tmp fit to find to then subtract the phase
    fit_tmp=fit_sine_poly(t=t, x=x_in, polyord=0, freqs=[f], err=dx)
    
    # extract fit
    V0=fit_tmp[0][0]
    A=fit_tmp[0][1]
    B=fit_tmp[0][2]
    C=np.sqrt(A**2+B**2)
    phi=-np.arctan2(B,A)
    t0=-(phi/w)
    
    # refit
    fit_IN=fit_sine_poly(t=t, x=x_in, polyord=0, freqs=[f], err=dx, t0=t0)
    fit_OUT=fit_sine_poly(t=t, x=x_out, polyord=0, freqs=[f], err=dx, t0=t0)
    return fit_IN[0][1:3],fit_OUT[0][1:3]
#2cm
for i in range(20):
    fname='c{}.csv'.format(i+1)
    fit_d0_IN[i//5,i%5],fit_d0_OUT[i//5,i%5]=sine_fits(fname,fs1[i//5])
#2.4cm
for i in range(20,40):
    fname='c{}.csv'.format(i+1)
    fit_d1_IN[(i-20)//5,(i-20)%5],fit_d1_OUT[(i-20)//5,(i-20)%5]=sine_fits(fname,fs1[(i-20)//5])
#4.3cm
for i in range(40,60):
    fname='c{}.csv'.format(i+1)
    fit_d2_IN[(i-40)//5,(i-40)%5],fit_d2_OUT[(i-40)//5,(i-40)%5]=sine_fits(fname,fs1[(i-40)//5])
#6.3cm
for i in range(60,80):
    fname='c{}.csv'.format(i+1)
    fit_d3_IN[(i-60)//5,(i-60)%5],fit_d3_OUT[(i-60)//5,(i-60)%5]=sine_fits(fname,fs1[(i-60)//5])
#9.3cm
for i in range(80,100):
    fname='c{}.csv'.format(i+1)
    fit_d4_IN[(i-80)//5,(i-80)%5],fit_d4_OUT[(i-80)//5,(i-80)%5]=sine_fits(fname,fs1[(i-80)//5])
#13.6cm
for i in range(100,115):
    fname='c{}.csv'.format(i+1)
    fit_d5_IN[(i-100)//5,(i-100)%5],fit_d5_OUT[(i-100)//5,(i-100)%5]=sine_fits(fname,fs1[1:4][(i-100)//5])
#20cm
for i in range(115,125):
    fname='c{}.csv'.format(i+1)
    fit_d6_IN[(i-115)//5,(i-115)%5],fit_d6_OUT[(i-115)//5,(i-115)%5]=sine_fits(fname,fs1[2:4][(i-115)//5])
#Mean-it
d0_IN=np.mean(fit_d0_IN,axis=1)
d0_OUT=np.mean(fit_d0_OUT,axis=1)
d1_IN=np.mean(fit_d1_IN,axis=1)
d1_OUT=np.mean(fit_d1_OUT,axis=1)
d2_IN=np.mean(fit_d2_IN,axis=1)
d2_OUT=np.mean(fit_d2_OUT,axis=1)
d3_IN=np.mean(fit_d3_IN,axis=1)
d3_OUT=np.mean(fit_d3_OUT,axis=1)
d4_IN=np.mean(fit_d4_IN,axis=1)
d4_OUT=np.mean(fit_d4_OUT,axis=1)
d5_IN=np.mean(fit_d5_IN,axis=1)
d5_OUT=np.mean(fit_d5_OUT,axis=1)
d6_IN=np.mean(fit_d6_IN,axis=1)
d6_OUT=np.mean(fit_d6_OUT,axis=1)
#Calculate Zeff
Zeff_d0=(d0_OUT[:,0]+1j*d0_OUT[:,1])/Gdiff_model(fs1*2*np.pi)/ \
        ( (d0_IN[:,0]+1j*d0_IN[:,1])/Rlim )
Zeff_d1=(d1_OUT[:,0]+1j*d1_OUT[:,1])/Gdiff_model(fs1*2*np.pi)/ \
        ( (d1_IN[:,0]+1j*d1_IN[:,1])/Rlim )
Zeff_d2=(d2_OUT[:,0]+1j*d2_OUT[:,1])/Gdiff_model(fs1*2*np.pi)/ \
        ( (d2_IN[:,0]+1j*d2_IN[:,1])/Rlim )
Zeff_d3=(d3_OUT[:,0]+1j*d3_OUT[:,1])/Gdiff_model(fs1*2*np.pi)/ \
        ( (d3_IN[:,0]+1j*d3_IN[:,1])/Rlim )
Zeff_d4=(d4_OUT[:,0]+1j*d4_OUT[:,1])/Gdiff_model(fs1*2*np.pi)/ \
        ( (d4_IN[:,0]+1j*d4_IN[:,1])/Rlim )
Zeff_d5=(d5_OUT[:,0]+1j*d5_OUT[:,1])/Gdiff_model(fs1[1:4]*2*np.pi)/ \
        ( (d5_IN[:,0]+1j*d5_IN[:,1])/Rlim )
Zeff_d6=(d6_OUT[:,0]+1j*d6_OUT[:,1])/Gdiff_model(fs1[2:4]*2*np.pi)/ \
        ( (d6_IN[:,0]+1j*d6_IN[:,1])/Rlim )
#Regressions
Regd0=linear_regression_AB(x=fs1*2*np.pi,y=d0_OUT[:,1],w=1)
Regd1=linear_regression_AB(x=fs1*2*np.pi,y=d1_OUT[:,1],w=1)
Regd2=linear_regression_AB(x=fs1*2*np.pi,y=d2_OUT[:,1],w=1)
Regd3=linear_regression_AB(x=fs1*2*np.pi,y=d3_OUT[:,1],w=1)
Regd4=linear_regression_AB(x=fs1*2*np.pi,y=d4_OUT[:,1],w=1)
Regd5=linear_regression_AB(x=fs1[1:4]*2*np.pi,y=d5_OUT[:,1],w=1)
Regd6=linear_regression_AB(x=fs1[2:4]*2*np.pi,y=d6_OUT[:,1],w=1)
#Plot-it
fig5=plt.figure()
fig5.suptitle('Frequency sweeps as function of distance')
ax51=fig5.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
ax52=fig5.add_axes([0.1, 0.1, 0.8, 0.4])
ax51.set_yscale('log')
ax51.set_xscale('log')
ax52.set_xscale('log')
ax51.set_ylabel('Zeff [Ω]')
ax52.set_ylabel('Phase [deg]')
ax52.set_xlabel('f [ s^-1]')
ax51.grid()
ax51.grid(b=True,which='minor',axis='both',alpha=0.5)
ax52.grid()
ax52.grid(b=True,which='minor',axis='both',alpha=0.5)

ax51.errorbar(x=fs1,y=np.abs(Zeff_d0),label='{}cm'.format(ds[0]*100),fmt='bo')
ax52.errorbar(x=fs1,y=np.angle(Zeff_d0)*180/np.pi,fmt='bo')
ax51.errorbar(x=fs1,y=np.abs(Zeff_d1),label='{}cm'.format(ds[1]*100),fmt='ro')
ax52.errorbar(x=fs1,y=np.angle(Zeff_d1)*180/np.pi,fmt='ro')
ax51.errorbar(x=fs1,y=np.abs(Zeff_d2),label='{}cm'.format(ds[2]*100),fmt='go')
ax52.errorbar(x=fs1,y=np.angle(Zeff_d2)*180/np.pi,fmt='go')
ax51.errorbar(x=fs1,y=np.abs(Zeff_d3),label='{}cm'.format(ds[3]*100),fmt='ko')
ax52.errorbar(x=fs1,y=np.angle(Zeff_d3)*180/np.pi,fmt='ko')
ax51.errorbar(x=fs1,y=np.abs(Zeff_d4),label='{}cm'.format(ds[4]*100),fmt='mo')
ax52.errorbar(x=fs1,y=np.angle(Zeff_d4)*180/np.pi,fmt='mo')
ax51.errorbar(x=fs1[1:4],y=np.abs(Zeff_d5),label='{:.2f}cm'.format(ds[5]*100),fmt='yo')
ax52.errorbar(x=fs1[1:4],y=np.angle(Zeff_d5)*180/np.pi,fmt='yo')
ax51.errorbar(x=fs1[2:4],y=np.abs(Zeff_d6),label='{}cm'.format(ds[6]*100),fmt='co')
ax52.errorbar(x=fs1[2:4],y=np.angle(Zeff_d6)*180/np.pi,fmt='co')

#plot models
ax51.errorbar(x=fs1,y=np.abs(Regd0[0]+Regd0[1]*fs1*2*np.pi),fmt='b-')
ax51.errorbar(x=fs1,y=np.abs(Regd1[0]+Regd1[1]*fs1*2*np.pi),fmt='r-')
ax51.errorbar(x=fs1,y=np.abs(Regd2[0]+Regd2[1]*fs1*2*np.pi),fmt='g-')
ax51.errorbar(x=fs1,y=np.abs(Regd3[0]+Regd3[1]*fs1*2*np.pi),fmt='k-')
ax51.errorbar(x=fs1,y=np.abs(Regd4[0]+Regd4[1]*fs1*2*np.pi),fmt='m-')
ax51.errorbar(x=fs1,y=np.abs(Regd5[0]+Regd5[1]*fs1*2*np.pi),fmt='y-')
ax51.errorbar(x=fs1,y=np.abs(Regd6[0]+Regd6[1]*fs1*2*np.pi),fmt='c-')

ax51.legend()

#%% distance sweep
f=200e3
arr=na([Zeff_d0[3],Zeff_d1[3],Zeff_d2[3],Zeff_d3[3],
        Zeff_d4[3],Zeff_d5[2],Zeff_d6[1]]).imag
M_RS=arr/(f*2*np.pi)
M_RS_log=np.log(M_RS)
ds_log=np.log(ds)
A1,B1,_,_=linear_regression_AB(x=ds_log,y=M_RS_log,w=1)

fig6=plt.figure()
fig6.suptitle('M_RS in function of distance @200kHz')
ax6=fig6.add_subplot(111)
ax6.set_ylabel('M_RS [H]')
ax6.set_xlabel('Distance [m]')
ax6.set_yscale('log')
ax6.set_xscale('log')
ax6.grid()
ax6.grid(which='minor',axis='x',alpha=0.5)
ax6.errorbar(x=ds,y=M_RS,fmt='bo')
ax6.errorbar(x=ds,y=np.exp(A1+B1*ds_log),fmt='b-')

#dipole approximation
iStmp=1
mu0=1.2566e-6
Area=np.pi*r**2
ms=N*iStmp*Area
Bz=mu0*2*ms/(4*np.pi*(ds)**3)/2
Bflux=N*Bz*Area
M=Bflux/iStmp
ax6.errorbar(x=ds,y=M,fmt='r-')
