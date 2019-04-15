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

freqs=[200,700,2000]

#%% Import data deltaR
fit_deltaR_IN=np.empty((6,5,3))
fit_deltaR_OUT=np.empty(fit_deltaR_IN.shape)

dfit_deltaR_IN=np.empty((6,5,3))
dfit_deltaR_OUT=np.empty(fit_deltaR_IN.shape)

for i in range(30):
    # import the dataframe
    fname='u{}.csv'.format(i+1)
    df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
    
    # extract the quantities
    t=df['x-axis']
    x_in=df['1']
    x_out=df['2']
    f=200
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
    
    # save
    fit_deltaR_IN[i//5,i%5]=fit_IN[0]
    fit_deltaR_OUT[i//5,i%5]=fit_OUT[0]
    dfit_deltaR_IN[i//5,i%5]=fit_IN[1]
    dfit_deltaR_OUT[i//5,i%5]=fit_OUT[1]

#%% Mean fits R

fit_deltaR_IN_mean=np.mean(fit_deltaR_IN,axis=1)[[0,2,1,3,4,5]] #invert because yes
fit_deltaR_OUT_mean=np.mean(fit_deltaR_OUT,axis=1)[[0,2,1,3,4,5]]

# take the std as uncertainty
dfit_deltaR_IN_mean=np.std(fit_deltaR_IN,axis=1)[[0,2,1,3,4,5]]
dfit_deltaR_OUT_mean=np.std(fit_deltaR_OUT,axis=1)[[0,2,1,3,4,5]]

#%% Import data deltaC
fit_deltaC_IN=np.empty((3,5,5,3))
fit_deltaC_OUT=np.empty(fit_deltaC_IN.shape)

dfit_deltaC_IN=np.empty(fit_deltaC_IN.shape)
dfit_deltaC_OUT=np.empty(fit_deltaC_IN.shape)

for i in range(60):
    # import the dataframe
    fname='v{}.csv'.format(i+1)
    df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
    
    # extract the quantities
    t=df['x-axis']
    x_in=df['1']
    x_out=df['2']
    f=freqs[i//20]
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
    
    # save
    fit_deltaC_IN[i//20,i%20//5+1,i%20%5]=fit_IN[0]
    fit_deltaC_OUT[i//20,i%20//5+1,i%20%5]=fit_OUT[0]
    dfit_deltaC_IN[i//20,i%20//5+1,i%20%5]=fit_IN[1]
    dfit_deltaC_OUT[i//20,i%20//5+1,i%20%5]=fit_OUT[1]
    
for i in range(15):
    # import the dataframe
    fname='w{}.csv'.format(i+1)
    df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
    
    # extract the quantities
    t=df['x-axis']
    x_in=df['1']
    x_out=df['2']
    f=freqs[i//5]
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
    
    # save
    fit_deltaC_IN[i//5,0,i%5]=fit_IN[0]
    fit_deltaC_OUT[i//5,0,i%5]=fit_OUT[0]
    dfit_deltaC_IN[i//5,0,i%5]=fit_IN[1]
    dfit_deltaC_OUT[i//5,0,i%5]=fit_OUT[1]

#%% Mean fits C

fit_deltaC_IN_mean=np.mean(fit_deltaC_IN,axis=2)
fit_deltaC_OUT_mean=np.mean(fit_deltaC_OUT,axis=2)

# take the std as uncertainty
dfit_deltaC_IN_mean=np.std(fit_deltaC_IN,axis=2)
dfit_deltaC_OUT_mean=np.std(fit_deltaC_OUT,axis=2)

#%% Plots
#Re_OUT=np.concatenate([fit_deltaR_OUT_mean[:,1],fit_deltaC_OUT_mean[0,:,1]])
#Im_OUT=-np.concatenate([fit_deltaR_OUT_mean[:,2],fit_deltaC_OUT_mean[0,:,2]])

deltaR_vout=np.array([fit_deltaR_OUT_mean[:,1],-fit_deltaR_OUT_mean[:,2]])
deltaC200_vout=np.array([fit_deltaC_OUT_mean[0,:,1],-fit_deltaC_OUT_mean[0,:,2]])
deltaC700_vout=np.array([fit_deltaC_OUT_mean[1,:,1],-fit_deltaC_OUT_mean[1,:,2]])
deltaC2000_vout=np.array([fit_deltaC_OUT_mean[2,:,1],-fit_deltaC_OUT_mean[2,:,2]])

#r_deltaR_OUT=np.sqrt(Re_deltaR_OUT**2+Im_deltaR_OUT**2)
#phi_deltaR_OUT=-np.arctan2(Im_deltaR_OUT,Re_deltaR_OUT)

fig_polar=plt.figure()
fig_polar.suptitle('δR,δC variations comparison')
ax_polar=fig_polar.add_subplot(111,)
ax_polar.errorbar(x=deltaR_vout[0],y=deltaR_vout[1],fmt='.b',label='δR 200Hz')
ax_polar.errorbar(x=deltaC200_vout[0],y=deltaC200_vout[1],fmt='.r',label='δC 200Hz')
ax_polar.errorbar(x=deltaC700_vout[0],y=deltaC700_vout[1],fmt='.m',label='δC 700Hz')
ax_polar.errorbar(x=deltaC2000_vout[0],y=deltaC2000_vout[1],fmt='.g',label='δC 2000Hz')
#ax_polar.plot(phi_deltaR_OUT,r_deltaR_OUT,'.')

ax_polar.set_xlabel('Vout Real [V]')
ax_polar.set_ylabel('Vout Imag [V]')

lim=60e-3
ax_polar.set_xlim(lim*np.array([-1,1]))
ax_polar.set_ylim(lim*np.array([-1,1]))
ax_polar.grid()
ax_polar.legend()

#%% Analysis
Rr=1001.5
R1=1568.5
R2=342.46
Rx_exp=Rr*R2/R1

Rx_DMM=216.88
Rs=na([Rx_DMM,par(Rx_DMM,1e5),par(Rx_DMM,1e5,1e5),par(Rx_DMM,1e5,1e5,1e5),
    par(Rx_DMM,1e5,1e5,1e5,1e5),par(Rx_DMM,1e5,1e5,1e5,1e5,1e5)])
Rsdelta=Rs-Rx_DMM
Cd=1e-9
Csdelta=Cd*np.arange(5)

Cs=lambda w: na([Rx_DMM,par(Rx_DMM,1/(1j*w*Cd)),par(Rx_DMM,1/(1j*w*Cd),1/(1j*w*Cd)),
              par(Rx_DMM,1/(1j*w*Cd),1/(1j*w*Cd),1/(1j*w*Cd)),
              par(Rx_DMM,1/(1j*w*Cd),1/(1j*w*Cd),1/(1j*w*Cd),1/(1j*w*Cd))])

# regression with y=a*x+b
#deltaRcoef=np.polyfit(x=Rs,y=deltaR_vout[0]+1j*deltaR_vout[1],deg=1)
#deltaC200coef=np.polyfit(x=Cs(2*np.pi*200),y=deltaC200_vout[0]+1j*deltaC200_vout[1],deg=1)
#deltaC700coef=np.polyfit(x=Cs(2*np.pi*700),y=deltaC700_vout[0]+1j*deltaC700_vout[1],deg=1)
#deltaC2000coef=np.polyfit(x=Cs(2*np.pi*2000),y=deltaC2000_vout[0]+1j*deltaC2000_vout[1],deg=1)
#ax_polar.plot( (deltaRcoef[1]+deltaRcoef[0]*np.array([Rs[0],Rs[-1]])).real ,
#              (deltaRcoef[1]+deltaRcoef[0]*np.array([Rs[0],Rs[-1]])).imag )

    
# regression with y=a*x
deltaRcoef=general_regression(F= Rsdelta.reshape((6,1)), 
                           y= deltaR_vout[0])[0][0] + 1j* \
           general_regression(F= Rsdelta.reshape((6,1)),
                           y= deltaR_vout[1])[0][0]
deltaC200coef=general_regression(F= Csdelta.reshape((5,1)), 
                           y= deltaC200_vout[0])[0][0] + 1j* \
           general_regression(F= Csdelta.reshape((5,1)),
                           y= deltaC200_vout[1])[0][0]
deltaC700coef=general_regression(F= Csdelta.reshape((5,1)), 
                           y= deltaC700_vout[0])[0][0] + 1j* \
           general_regression(F= Csdelta.reshape((5,1)),
                           y= deltaC700_vout[1])[0][0]
deltaC2000coef=general_regression(F= Csdelta.reshape((5,1)), 
                           y= deltaC2000_vout[0])[0][0] + 1j* \
           general_regression(F= Csdelta.reshape((5,1)),
                           y= deltaC2000_vout[1])[0][0]

#regression unc only what is needed
ddeltaRcoef=general_regression(F= Rsdelta.reshape((6,1)), 
                           y= deltaR_vout[0],dy=dfit_deltaR_OUT_mean[:,1]*np.sqrt(3343))[1][0]
ddeltaC200coef=general_regression(F= Csdelta.reshape((5,1)),
                           y= deltaC200_vout[1],dy=dfit_deltaC_OUT_mean[0,:,2]*np.sqrt(790))[1][0]
ddeltaC700coef=general_regression(F= Csdelta.reshape((5,1)),
                           y= deltaC700_vout[1],dy=dfit_deltaC_OUT_mean[1,:,2])[1][0]
ddeltaC2000coef=general_regression(F= Csdelta.reshape((5,1)),
                           y= deltaC2000_vout[1],dy=dfit_deltaC_OUT_mean[2,:,2]*np.sqrt(22981))[1][0]

print('d(Re[Vout])/d(deltaR)=',ufloat(deltaRcoef.real,ddeltaRcoef))
print('d(Im[Vout])/d(deltaC)=',ufloat(deltaC200coef.imag,ddeltaC200coef),'@200Hz')
print('d(Im[Vout])/d(deltaC)=',ufloat(deltaC700coef.imag,ddeltaC700coef),'@700Hz')
print('d(Im[Vout])/d(deltaC)=',ufloat(deltaC2000coef.imag,ddeltaC2000coef),'@2000Hz')

print('Chi2red deltaR=',chi2red(deltaR_vout[0],dfit_deltaR_OUT_mean[:,1],
                                deltaRcoef.real*Rsdelta,1))
print('Chi2red deltaC200=',chi2red(deltaC200_vout[1],dfit_deltaC_OUT_mean[0,:,2],
                                   deltaC200coef.imag*Csdelta,1))
print('Chi2red deltaC700=',chi2red(deltaC700_vout[1],dfit_deltaC_OUT_mean[1,:,2],
                                   deltaC700coef.imag*Csdelta,1))
print('Chi2red deltaC2000=',chi2red(deltaC2000_vout[1],dfit_deltaC_OUT_mean[2,:,2],
                                   deltaC2000coef.imag*Csdelta,1))
print('Recalculate for chi2red=1') #metti apposto il chi2

fig1=plt.figure()
fig1.suptitle('δR @200Hz')
ax1=fig1.add_subplot(111)
ax1.errorbar(x=Rsdelta,y=deltaR_vout[0],fmt='xb',yerr=dfit_deltaR_OUT_mean[:,1]*np.sqrt(3343))
ax1.errorbar(x=Rsdelta,y=deltaR_vout[1],fmt='xr',yerr=dfit_deltaR_OUT_mean[:,2])
ax1.errorbar(x=Rsdelta,y=deltaRcoef.real*Rsdelta,fmt=',b-',label='Real')
ax1.errorbar(x=Rsdelta,y=deltaRcoef.imag*Rsdelta,fmt=',r-',label='Imag')
ax1.set_ylabel('Vout [V]')
ax1.set_xlabel('δR=R-Rx [Ω]')
ax1.legend()


fig2=plt.figure()
fig2.suptitle('δC @200Hz')
ax2=fig2.add_subplot(111)
ax2.errorbar(x=Csdelta,y=deltaC200_vout[0],fmt='xb',yerr=dfit_deltaC_OUT_mean[0,:,1])
ax2.errorbar(x=Csdelta,y=deltaC200_vout[1],fmt='xr',yerr=dfit_deltaC_OUT_mean[0,:,2])
ax2.errorbar(x=Csdelta,y=deltaC200coef.real*Csdelta,fmt=',b-',label='Real')
ax2.errorbar(x=Csdelta,y=deltaC200coef.imag*Csdelta,fmt=',r-',label='Imag')
ax2.set_ylabel('Vout [V]')
ax2.set_xlabel('δC [F]')
ax2.legend()

fig3=plt.figure()
fig3.suptitle('δC @700Hz')
ax3=fig3.add_subplot(111)
ax3.errorbar(x=Csdelta,y=deltaC700_vout[0],fmt='xb',yerr=dfit_deltaC_OUT_mean[1,:,1])
ax3.errorbar(x=Csdelta,y=deltaC700_vout[1],fmt='xr',yerr=dfit_deltaC_OUT_mean[1,:,2]*np.sqrt(790))
ax3.errorbar(x=Csdelta,y=deltaC700coef.real*Csdelta,fmt=',b-',label='Real')
ax3.errorbar(x=Csdelta,y=deltaC700coef.imag*Csdelta,fmt=',r-',label='Imag')
ax3.set_ylabel('Vout [V]')
ax3.set_xlabel('δC [F]')
ax3.legend()

fig4=plt.figure()
fig4.suptitle('δC @2000Hz')
ax4=fig4.add_subplot(111)
ax4.errorbar(x=Csdelta,y=deltaC2000_vout[0],fmt='xb',yerr=dfit_deltaC_OUT_mean[2,:,1])
ax4.errorbar(x=Csdelta,y=deltaC2000_vout[1],fmt='xr',yerr=dfit_deltaC_OUT_mean[2,:,2]*np.sqrt(22981))
ax4.errorbar(x=Csdelta,y=deltaC2000coef.real*Csdelta,fmt=',b-',label='Real')
ax4.errorbar(x=Csdelta,y=deltaC2000coef.imag*Csdelta,fmt=',r-',label='Imag')
ax4.set_ylabel('Vout [V]')
ax4.set_xlabel('δC [F]')
ax4.legend()



#%%
Cosc=110e-12
Rosc=1e6
Cout=100e-9
Gdiff=35.65
Gdiff_model=lambda w: Gdiff * par(1/(1j*w*Cosc),Rosc)/ \
            (par(1/(1j*w*Cosc),Rosc)+Rc+1/(1j*w*Cout))

Re=100
re=40.226629
beta=350
Zbridge_out=par(Rx_DMM,Rr)+par(R1,R2)
Zamp_in=2*beta*(Re+re)

VoutdeltaR=lambda w: fit_deltaR_IN_mean[:,1]*Rr/(Rx_DMM+Rr)**2 * \
                (Rsdelta)*Zamp_in/(Zamp_in+Zbridge_out)*Gdiff_model(w)
VoutdeltaC=lambda w: np.mean(fit_deltaC_IN_mean[:,:,1],axis=0)*Rr/(Rx_DMM+Rr)**2 * \
                (Rs[0]**2*(-1j*w*Csdelta))*Zamp_in/(Zamp_in+Zbridge_out)*Gdiff_model(w)

sigmadeltaR=np.max(dfit_deltaR_OUT_mean[:,1]*np.sqrt(3343)/deltaRcoef.real)
sigmadeltaC200=np.max(dfit_deltaC_OUT_mean[0,:,2]/deltaC200coef.imag)
sigmadeltaC700=np.max(dfit_deltaC_OUT_mean[1,:,2]*np.sqrt(790)/deltaC700coef.imag)
sigmadeltaC2000=np.max(dfit_deltaC_OUT_mean[2,:,2]*np.sqrt(22981)/deltaC2000coef.imag)
print('Sensibility:')
print('sigmadeltaR=',sigmadeltaR)
print('sigmadeltaC200=',sigmadeltaC200)
print('sigmadeltaC700',sigmadeltaC700)
print('sigmadeltaC2000',sigmadeltaC2000)

#%% Plot balanced bridge
fig5=plt.figure()
fig5.suptitle('Bridge de-balancing')
ax5=fig5.add_subplot(111)
# import the dataframe
fname='u{}.csv'.format(0+1)
df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
# extract the quantities
t=df['x-axis']
x_in=df['1']
x_out=df['2']
ax5.plot(t,x_out,'-',label='δR=0')
# import the dataframe
fname='u{}.csv'.format(10+1)
df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
# extract the quantities
t=df['x-axis']
x_in=df['1']
x_out=df['2']
ax5.plot(t,x_out,'-',label='R+δR')
# import the dataframe
fname='u{}.csv'.format(5+1)
df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
# extract the quantities
t=df['x-axis']
x_in=df['1']
x_out=df['2']
ax5.plot(t,x_out,'-',label='R+2δR')

# import the dataframe
fname='u{}.csv'.format(15+1)
df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
# extract the quantities
t=df['x-axis']
x_in=df['1']
x_out=df['2']
ax5.plot(t,x_out,'-',label='R+3δR')
# import the dataframe
fname='u{}.csv'.format(20+1)
df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
# extract the quantities
t=df['x-axis']
x_in=df['1']
x_out=df['2']
ax5.plot(t,x_out,'-',label='R+4δR')
# import the dataframe
fname='u{}.csv'.format(25+1)
df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
# extract the quantities
t=df['x-axis']
x_in=df['1']
x_out=df['2']
ax5.plot(t,x_out,'-',label='R+5δR')

ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Vout [V]')
ax5.legend()
#%%
fig7=plt.figure()
fig7.suptitle('Balanced bridge')
ax7=fig7.add_subplot(111)
ax7.set_ylabel('Vout [V]')
ax7.set_xlabel('Time [s]')
for i in range(5):
    # import the dataframe
    fname='u{}.csv'.format(i+1)
    df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
    
    # extract the quantities
    t=df['x-axis']
    x_in=df['1']
    x_out=df['2']
    f=200
    w=2*np.pi*f
    dx=0.03*np.max(x_in)
    
    # first tmp fit to find to then subtract the phase
    fit_tmp=fit_sine_poly(t=t, x=x_out, polyord=0, freqs=[f], err=dx)
    
    # extract fit
    V0=fit_tmp[0][0]
    A=fit_tmp[0][1]
    B=fit_tmp[0][2]
    C=np.sqrt(A**2+B**2)
    phi=-np.arctan2(B,A)
    t0=-(phi/w)
    
    # plot
    ax7.errorbar(x=t-t0,y=x_out-V0,fmt='-')

#%%
fig6=plt.figure()
fig6.suptitle('Anomaly')
ax6=fig6.add_subplot(111)
for i in range(6*5):
    # import the dataframe
    fname='u{}.csv'.format(i+1)
    df=pd.read_csv('data/osccapt1/'+fname,skiprows=[1],header=0)
    # extract the quantities
    t=df['x-axis']
    x_in=df['1']
    x_out=df['2']
    ax6.plot(t,x_out,'{}-'.format(['r','g','b','c','m','y'][i//5]),
             label=('R+{}δR'.format(i//5) if i%5==0 else '_nolegend_'))
    
ax6.set_xlabel('Time [s]')
ax6.set_ylabel('Vout [V]')
ax6.legend()



