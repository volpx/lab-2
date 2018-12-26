#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:51:58 2018

@author: volpe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,4)

# diode data
df=pd.read_csv('data5/D1.csv',header=0).values
diode=pd.DataFrame({'V':df[:,2],'I':df[:,1]*1e-3})

def diode_i(v):
    '''
        Diode characteristic
    '''
    if v<=diode['V'][0]:
        return 0
    j=1
    while j<diode['V'].size-1 and not v < diode['V'][j]:
        j+=1
    return diode['I'][j-1]+(diode['I'][j] - diode['I'][j-1])/(diode['V'][j] - diode['V'][j-1]) * (v-diode['V'][j-1])

def diode_v(i):
    '''
        Diode characteristic
    '''
    j=1
    while j<diode['I'].size-1 and not i < diode['I'][j]:
        j+=1
    res=diode['V'][j-1]+(diode['V'][j] - diode['V'][j-1])/(diode['I'][j] - diode['I'][j-1]) * (i-diode['I'][j-1])
    return res if res > 0 else 0

# zener data
df=pd.read_csv('data5/Z.csv',header=0).values
dzener=pd.DataFrame({'V':df[:,2],'I':df[:,1]*1e-3})

Arz_log=-2.49470302
Brz_log=-1.20302689

def rz(i):
    return np.exp(Brz_log*np.log(i)+Arz_log)

def zener_v(i):
    '''
        Zener characteristic
    '''
    j=1
    while j<dzener['I'].size-1 and not i < dzener['I'][j]:
        j+=1
    res=dzener['V'][j-1]+(dzener['V'][j] - dzener['V'][j-1])/(dzener['I'][j] - dzener['I'][j-1]) * (i-dzener['I'][j-1])
    return res if res > 0 else 0

def zener_i(v):
    '''
        Zener characteristic
    '''
    if v<=dzener['V'][0]:
        return 0
    j=1
    while j<dzener['V'].size-1 and not v < dzener['V'][j]:
        j+=1
    res=dzener['I'][j-1]+(dzener['I'][j] - dzener['I'][j-1])/(dzener['V'][j] - dzener['V'][j-1]) * (v-dzener['V'][j-1])
    return res if res > 0 else 0

def FBR_sim(vin_rms,Rl):
    #set timings
    x=np.linspace(0,1,1e6)

    #invent vin
    vin=vin_rms*np.sqrt(2)*np.cos(2*np.pi*50*x)

    vout=np.empty((x.size,))
    vout[0]=vin[0]

    #sim
    for i in range(1,x.size):
        if vin[i-1]-2*diode_v()-4:
            # diodes conduct
            pass

#%% FBR

# sperimental data
df1=pd.read_csv('data6/FBR.csv')
vout_max_e=df1['VoutRMS']+0.5*df1['Pk-Pk out']
dvout_max_e=vout_max_e/.8*0.03
vout_pp_e=df1['Pk-Pk out']
dvout_pp_e=vout_pp_e/.8*0.03*np.sqrt(2)
vin_max=df1['VinRMS']*1.365 #factor from SM #*np.sqrt(2)

Rl=df1['Rl']

# model
i_max_e=vout_max_e/Rl
vout_max_m=vin_max-2*np.vectorize(diode_v)(i_max_e)
#vout_pp_m=i_max_e/100/220e-6
vout_pp_m=vout_max_m/Rl/100/220e-6


fig1=plt.figure()
fig1.suptitle('In/Out characteristic')
ax1=fig1.add_subplot(111)


ax1.errorbar(x=Rl,y=vout_max_e,yerr=dvout_max_e,fmt='b.',label='Vout max')
ax1.plot(Rl,vin_max,'r.',label='Vin max')
ax1.plot(Rl,vout_max_m,'g.',label='Vout max model')
ax1.set_xscale('log')
ax1.set_xlabel('R load [Ω]')
ax1.set_ylabel('V max [V]')
ax1.grid()
ax1.legend()

#fig1.savefig('report/fig1.pdf')


fig2=plt.figure()
fig2.suptitle('Ripple out')
ax2=fig2.add_subplot(111)

ax2.errorbar(x=Rl,y=vout_pp_e,yerr=dvout_pp_e,fmt='b.',label='Vout pp')
ax2.plot(Rl,vout_pp_m,'g.',label='Vout pp model')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('R load [Ω]')
ax2.set_ylabel('V pp [V]')
ax2.grid(which='major')
ax2.grid(which='minor',alpha=.2)

ax2.legend()
#fig2.savefig('report/fig2.pdf')

#fig3=plt.figure()
#fig3.suptitle('Ripple ratio')
#ax3=fig3.add_subplot(111)
#
#ax3.plot(Rl,vout_pp_e/vout_max_e,'b.')
#ax3.set_xscale('log')
#ax3.set_yscale('log')
#ax3.set_xlabel('R load [Ω]')
#ax3.set_ylabel('Vout/Vpp []')
#ax3.grid()

#%% FBRD

# sperimental data
df2=pd.read_csv('data6/FBRD.csv')

vc_max_e=df2['Vc MAX']
dvc_max_e=vc_max_e/.8*0.03
vc_pp_e=df2['Vc P-P']
dvc_pp_e=vc_pp_e/.8*0.03*np.sqrt(2)

vout_max_e=df2['Vout MAX']
dvout_max_e=vout_max_e/.8*0.03
vout_pp_e=df2['Vout P-P']
dvout_pp_e=vout_pp_e/.8*0.03*np.sqrt(2)

Rl=df2['Rl']

# model
iout_max=vout_max_e/Rl
iz_max=np.vectorize(zener_i)(vout_max_e)
ir_max=iout_max+iz_max
vout_max_m=vc_max_e-ir_max*100

vc_pp_m=ir_max/100/220e-6
vout_pp_m=vc_pp_m*np.vectorize(rz)(iz_max)/(np.vectorize(rz)(iz_max)+100*(1+np.vectorize(rz)(iz_max)/Rl))


fig4=plt.figure()
fig4.suptitle('Diode filter characteristic')
ax4=fig4.add_subplot(111)


ax4.errorbar(x=Rl,y=vout_max_e,yerr=dvout_max_e,fmt='b.',label='Vout max')
ax4.errorbar(x=Rl,y=vc_max_e,yerr=dvc_max_e,fmt='g.',label='Vc max')
ax4.plot(Rl,vout_max_m,'r.:',label='Vout max model')
ax4.set_xscale('log')
ax4.set_xlabel('R load [Ω]')
ax4.set_ylabel('V max [V]')
ax4.grid()
ax4.legend()

#fig4.savefig('report/fig4.pdf')


fig5=plt.figure()
fig5.suptitle('Ripple out')
ax5=fig5.add_subplot(111)

ax5.errorbar(x=Rl,y=vout_pp_e,yerr=dvout_pp_e,fmt='b.',label='Vout pp')
ax5.plot(Rl,vout_pp_m,'r.:',label='Vout pp model')
ax5.errorbar(x=Rl,y=vc_pp_e,yerr=dvc_pp_e,fmt='g.',label='Vc pp')
ax5.plot(Rl,vc_pp_m,'c.:',label='Vc pp model')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlabel('R load [Ω]')
ax5.set_ylabel('V pp [V]')
ax5.grid(which='major',alpha=1)
ax5.grid(which='minor',alpha=.2)
ax5.legend()

#fig5.savefig('report/fig5.pdf')

#fig3=plt.figure()
#fig3.suptitle('Ripple ratio')
#ax3=fig3.add_subplot(111)
#
#ax3.plot(Rl,vout_pp_e/vout_max_e,'b.')
#ax3.set_xscale('log')
#ax3.set_yscale('log')
#ax3.set_xlabel('R load [Ω]')
#ax3.set_ylabel('Vout/Vpp []')
#ax3.grid()

#%% Rout
vout_ca=np.max(vout_max_e)-0.5*vout_pp_e[vout_pp_e.size-1]
Rout=(vout_ca-(vout_max_e-0.5*vout_pp_e))/((vout_max_e-0.5*vout_pp_e)/Rl)

fig6=plt.figure()
fig6.suptitle('Rout')
ax6=fig6.add_subplot(111)

ax6.plot(Rl,Rout,'b.')
ax6.set_xscale('log')
ax6.set_ylabel('Rout [Ω]')
ax6.set_xlabel('R load [Ω]')
ax6.grid(which='major',alpha=1)
ax6.grid(which='minor',axis='x',alpha=.2)

#fig6.savefig('report/fig6.pdf')

#%% Efficiency

pout=vout_max_e**2/Rl

pz=iz_max*vout_max_e
pr=(vc_max_e-vout_max_e)**2/100
pd=pr + pz

fig7 = plt.figure()
fig7.suptitle('Efficiency')
ax7 = fig7.add_subplot(111)

ax7.plot(Rl,pout,'b.:',label='Pout')
ax7.plot(Rl,pz,'r.:',label='P z')
ax7.plot(Rl,pr,'m.:',label='P r')
ax7.plot(Rl,pd,'g.:',label='P d')
ax7b=ax7.twinx()
ax7b.plot(Rl,pout/(pout+pd),'k.:',label='efficiency')

ax7.set_xscale('log')
ax7.set_yscale('log')
ax7.set_xlabel('R load [Ω]')
ax7.set_ylabel('Power [W]')
ax7b.set_ylabel('efficiency []')
ax7.plot(np.nan, 'k.:', label = 'efficiency')
ax7.legend()

ax7.grid()

#fig7.savefig('report/fig7.pdf')

