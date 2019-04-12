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

from functions import *

# %% w/o CC gen
# import dataframe for diffamp w/o CC gen below
df1=pd.read_csv("data/diffamp.csv")

# calculate gains
Gdiff1=df1['outDIFF']/df1['in']
dGdiff1=np.sqrt(((df1['outDIFF']*0.015)*(1/df1['in']))**2 + 
    ((df1['in']*0.015)*(df1['outDIFF']/df1['in']**2)))
Gcm1=df1['outCM']/df1['in']
dGcm1=(df1['outCM']*0.015)*(1/df1['in']) + (df1['in']*0.015)*(df1['outCM']/df1['in']**2)

# calculate phases
Pdiff1=-df1['phDIFF']
dPdiff1=3 /10/df1['freq'] *80*1e-4* np.ones(Gdiff1.shape)
Pcm1=-df1['phCM']
dPcm1=3 /10/df1['freq'] *80*1e-4* np.ones(Gdiff1.shape)

##adjust some phases
#Pcm1.values[0:3]=Pcm1.values[0:3]-360

# bodeplot
fig_bode1,(_,_) = bode_plot(x=df1['freq'], H=np.vstack([Gdiff1,Pdiff1]).T, dB=False,
                          Herr=[dGdiff1,dPdiff1], err=True, title='Differential amp. w/o CC source')
fig_bode1,(ax_top1,ax_bot1) = bode_plot(x=df1['freq'], H=np.vstack([Gcm1,Pcm1]).T, fmt='r.', dB=False,
                          Herr=[dGcm1,dPcm1], err=True, ext_fig=fig_bode1)
ax_top1.set_ylabel('G []')
ax_top1.set_yscale('log')
ax_bot1.set_ylabel('φ [deg]')
ax_top1.grid(b=True, which='minor',axis='x', alpha=.5)
ax_bot1.grid(b=True, which='minor',axis='x', alpha=.5)

# %% w/ CC gen
# import dataframe for diffamp w/o CC gen below
df2=pd.read_csv('data/diffamp_ccg.csv')
df3=pd.read_csv('data/diffamp_ccg_more.csv')

# calculate gains
Gdiff2=df2['outDIFF']/df1['in']
dGdiff2=np.sqrt(((df2['outDIFF']*0.015)*(1/df2['in']))**2 + 
    ((df2['in']*0.015)*(df2['outDIFF']/df2['in']**2)))
Gcm2=df3['outCM']/df3['in']
dGcm2=(df3['outCM']*0.015)*(1/df3['in']) + (df3['in']*0.015)*(df3['outCM']/df3['in']**2)

# calculate phases
Pdiff2=-df2['phDIFF']
dPdiff2=3 /10/df1['freq'] *80*1e-4* np.ones(Gdiff2.shape)
Pcm2=-df3['phCM']
dPcm2=3 /10/df1['freq'] *80*1e-4* np.ones(Gdiff2.shape)

##adjust some phases

# bodeplot
fig_bode2,(_,_) = bode_plot(x=df1['freq'], H=np.vstack([Gdiff2,Pdiff2]).T, dB=False,
                          Herr=[dGdiff2,dPdiff2], err=True, title='Differential amp. w/ CC source')
fig_bode2,(ax_top2,ax_bot2) = bode_plot(x=df1['freq'], H=np.vstack([Gcm2,Pcm2]).T, fmt='r.', dB=False,
                          Herr=[dGcm2,dPcm2], err=True, ext_fig=fig_bode2)
##remove gibberish
ax_bot2.get_lines()[1].remove()

ax_top2.set_ylabel('G []')
ax_top2.set_yscale('log')
ax_bot2.set_ylabel('φ [deg]')
ax_top2.grid(b=True, which='minor',axis='x', alpha=.5)
ax_bot2.grid(b=True, which='minor',axis='x', alpha=.5)

# %% comparison w/ w/o gen

fig_bode3,(_,_) = bode_plot(x=df1['freq'], H=np.vstack([Gdiff1,Pdiff1]).T, dB=False,
                          Herr=[dGdiff1,dPdiff1], err=True, title='Gain differential amp. comparison')
fig_bode3,(ax_top3,ax_bot3) = bode_plot(x=df1['freq'], H=np.vstack([Gdiff2,Pdiff2]).T, fmt='r.', dB=False,
                          Herr=[dGdiff2,dPdiff2], err=True, ext_fig=fig_bode3)

ax_top3.set_ylabel('G []')
ax_top3.set_yscale('log')
ax_bot3.set_ylabel('φ [deg]')
ax_top3.grid(b=True, which='minor',axis='x', alpha=.5)
ax_bot3.grid(b=True, which='minor',axis='x', alpha=.5)
fig_bode4,(_,_) = bode_plot(x=df1['freq'], H=np.vstack([Gcm1,Pcm1]).T, dB=False,
                          Herr=[dGcm1,dPcm1], err=True, title='Gain CM differential amp. comparison')
fig_bode4,(ax_top4,ax_bot4) = bode_plot(x=df1['freq'], H=np.vstack([Gcm2,Pcm2]).T, fmt='r.', dB=False,
                          Herr=[dGcm2,dPcm2], err=True, ext_fig=fig_bode4)

ax_top4.set_ylabel('G []')
ax_top4.set_yscale('log')
ax_bot4.set_ylabel('φ [deg]')
ax_top4.grid(b=True, which='minor',axis='x', alpha=.5)
ax_bot4.grid(b=True, which='minor',axis='x', alpha=.5)

#%%
##circuit params
Rc=1e4
Re=100
R1=4.8e3

Gdiff=np.max(Gdiff2)
Gcm=np.max(Gcm2)
re_exp=(Rc-2*Gdiff2*Re)/2/Gdiff2

#Rs_mod=Va/Vt*R1
Zs_exp=-Rc/2/(Gcm2*np.exp(1j*Pcm2))
Zs_RE_exp=np.real(Zs_exp)
Zs_IM_exp=np.imag(Zs_exp)

Zs_mod=par( 2.8e7 ,1/(1j*df1['freq']*2*np.pi* 2e-12 ))

fig_bode5,(ax_top5,ax_bot5) = bode_plot(x=df1['freq'], H=Zs_exp, dB=False,
                              title='Zs as a function of frequency', y_label='Zs')
fig_bode5,(_,_) = bode_plot(x=df1['freq'], H=Zs_mod, fmt='r-', dB=False,
                           ext_fig=fig_bode5)
ax_top5.grid(b=True, which='minor',axis='x', alpha=.5)
ax_bot5.grid(b=True, which='minor',axis='x', alpha=.5)

#%% comparison with model of the one w/ gen
Cosc=110e-12
Rosc=1e6
Cout=100e-9
Gdiff_model=lambda w: Gdiff * par(1/(1j*w*Cosc),Rosc)/ \
            (par(1/(1j*w*Cosc),Rosc)+Rc+1/(1j*w*Cout))

fig_bode6,(_,_) = bode_plot(x=df1['freq'], H=np.vstack([Gdiff2,Pdiff2]).T,
                          Herr=[dGdiff2,dPdiff2], err=True, title='Gain differential amp. model')
fig_bode6,(ax_top6,ax_bot6) = bode_plot(x=df1['freq'], H=Gdiff_model(2*np.pi*df1['freq']), fmt=',r-', dB=False,
                                        ext_fig=fig_bode6)

ax_top6.set_ylabel('G []')
ax_top6.set_yscale('log')
ax_bot6.set_ylabel('φ [deg]')
ax_top6.grid(b=True, which='minor',axis='both', alpha=.5)
ax_bot6.grid(b=True, which='minor',axis='x', alpha=.5)


