#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:57:32 2018

@author: volpe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

#array of colors to use for plots
colors=["g","r","b","c"]
#import my class and functions
from functions import DataXY,w_mean

# coverage factor
k=2
#parallel and series resistances of ICE testers
R_par_10V=2e5
R_par_2V=4e4
R_ser_5mA=63.479143
R_ser_500uA=588.8
#nominal and DMM resistance values
R_n=4.7e3
dR_n=235
R_DMM=4596
dR_DMM=0.46

#make an array of DataFrame s with imported data fro my csv files
data=[pd.read_csv("data/AmpMon10V5mA.csv"),
         pd.read_csv("data/AmpMon2V0.5mA.csv"),
         pd.read_csv("data/AmpVal10V5mA.csv"),
         pd.read_csv("data/AmpVal2V0.5mA.csv")]

#convert in standart si units
for df in data:
    df["mA"]*=1e-3
    df.columns=["V","A"]

# array of full-scale values for each data series
data_fs=np.array([[10,5e-3],[2,500e-6],[10,5e-3],[2,500e-6]])
# array of resolution uncertainties
data_Delta=np.array([[0.2,100e-6],[0.04,10e-6],[0.2,100e-6],[0.04,10e-6]])
# get the standard deviation for uncertainties
data_sigma=data_Delta/np.sqrt(12)

# make an array of DataXY classes, one for each data serie
dataseries=[DataXY(data[0]["V"][0:],data[0]["A"][0:],\
                   data_sigma[0][0],data_sigma[0][1],\
                   x_label="V",y_label="A",\
                   name="Amperometro a Monte (FS: 10V/5mA)"),\
            DataXY(data[1]["V"],data[1]["A"],\
                   data_sigma[1][0],data_sigma[1][1],\
                   x_label="V",y_label="A",\
                   name="Amperometro a Monte (FS: 2V/500uA)"),\
            DataXY(data[2]["V"],data[2]["A"],\
                   data_sigma[2][0],data_sigma[2][1],\
                   x_label="V",y_label="A",\
                   name="Amperometro a Valle (FS: 10V/5mA)"),\
            DataXY(data[3]["V"],data[3]["A"],\
                   data_sigma[3][0],data_sigma[3][1],\
                   x_label="V",y_label="A",\
                   name="Amperometro a Valle (FS: 2V/500uA)"),\
            ]

# get fit plot and save it
dataseries[0].getFitPlot()[0].savefig("data/AmpMon10V5mA.pdf",bbox_inches="tight")
dataseries[1].getFitPlot()[0].savefig("data/AmpMon2V500uA.pdf",bbox_inches="tight")
dataseries[2].getFitPlot()[0].savefig("data/AmpVal10V5mA.pdf",bbox_inches="tight")
dataseries[3].getFitPlot()[0].savefig("data/AmpVal2V500uA.pdf",bbox_inches="tight")

# make plot with all datasets
# create a figure
fig_all=plt.figure()
# put title
fig_all.suptitle("Confronto datasets")
#get new ax
ax_all=fig_all.add_subplot(1,1,1)
# plot in the ax the errorbar and the fit
for i,data in enumerate(dataseries):
    ax_all.errorbar(data.x, data.y, yerr=data.dy, xerr=data.dx, fmt=colors[i]+".")
    ax_all.plot([0, data.x.max()], data.getModel(x=np.array([0, data.x.max()])), colors[i]+",-", label=data.name)
#put grid
ax_all.grid()
# put axis labels
ax_all.set_ylabel("A")
ax_all.set_xlabel("V")
# put legend on the top left
ax_all.legend(loc="upper left",bbox_to_anchor=(-.01,1.01))
# format ticks with scientific notations
ax_all.ticklabel_format(style="sci",scilimits=(0,0))
# save figure
fig_all.savefig("data/FigAll.pdf",bbox_inches="tight")

## Summary
# create two uninitialized array of size=(4,)
R_x=np.empty(4)
dR_x=np.empty(4)

# set index 
i=0
# print preformatted DataXY informations
dataseries[i].prettyPrint(ddof=2,dy_prop=True)
# get linear regression parameters
A,B,dA,dB = dataseries[i].getLinearRegressionAB()
# save corresponding R_x in the R_x[] array with its uncertainty
R_x[i]=1/(B - 1/R_par_10V)
dR_x[i]=np.abs(-1/(B-1/R_par_10V)**2) * dB
# print value and uncertainty
print("Rx =",ufloat(R_x[i],dR_x[i]))

i=1
dataseries[i].prettyPrint(ddof=2,dy_prop=True)
A,B,dA,dB = dataseries[i].getLinearRegressionAB()
R_x[i]=1/(B - 1/R_par_2V)
dR_x[i]=np.abs(-1/(B-1/R_par_2V)**2) * dB
print("Rx =",ufloat(R_x[i],dR_x[i]) )

i=2
dataseries[i].prettyPrint(ddof=2,dy_prop=True)
A,B,dA,dB = dataseries[i].getLinearRegressionAB()
R_x[i]=1/B - R_ser_5mA
dR_x[i]=np.abs(1/B**2) * dB
print("Rx =",ufloat(R_x[i],dR_x[i]) )

i=3
dataseries[i].prettyPrint(ddof=2,dy_prop=True)
A,B,dA,dB = dataseries[i].getLinearRegressionAB()
R_x[i]=1/B - R_ser_500uA
dR_x[i]=np.abs(1/B**2) * dB
print("Rx =",ufloat(R_x[i],dR_x[i]) )

#another blank line
print()

## Weighted mean
# get weighted mean
R_w,dR_w=w_mean(R_x,1/dR_x**2)
# print weighted mean
print("Weighted mean=",ufloat(R_w,dR_w))


## Compatibility of Rxs
fig_comp=plt.figure()
ax_comp=fig_comp.add_subplot(1,1,1)

for j,i in enumerate([2,0,3,1]):
    # plot an undefinitely long vertical line 
    ax_comp.axvline(x=R_x[i],color=colors[i],label=dataseries[i].name)
    # make a vertical span with width equal to k*sigma for each side
    ax_comp.axvspan(xmin=R_x[i]-k*dR_x[i], xmax=R_x[i]+k*dR_x[i], ymin=0.8-.2*j, ymax=1-0.2*j, color=colors[i],alpha=0.2)
# also plot R_n and R_DMM in the same plot
ax_comp.axvline(x=R_n,color="#464a45",label="R_n")
ax_comp.axvspan(xmin=R_n-dR_n,xmax=R_n+dR_n,color="#464a45",alpha=0.2)
ax_comp.axvline(x=R_DMM,color="#EEEE00",label="R_DMM")
ax_comp.axvspan(xmin=R_DMM-k*dR_DMM, xmax=R_DMM+k*dR_DMM, color="#EEEE00", alpha=0.2)
ax_comp.axvline(x=R_w,color="#ff8000",label="R_w")
ax_comp.axvspan(xmin=R_w-k*dR_w, xmax=R_w+k*dR_w, ymin=0, ymax=0.2, color="#ff8000",alpha=0.2)
ax_comp.set_xlabel(u"R [Ω]")
# disable ticks on y axis
ax_comp.set_yticks([])
ax_comp.legend(loc="center left",bbox_to_anchor=(1.03,.5))
fig_comp.suptitle("Comparazione Rx")
fig_comp.savefig("data/fig_comp.pdf",bbox_inches="tight")













