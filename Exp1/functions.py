#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:07:26 2018

@author: volpe
"""

## Library of functions
import numpy as np
import matplotlib.pyplot as plt

def w_mean(data,weigth=None):
    if weigth.any():
        mean=np.sum(data*weigth)/np.sum(weigth)
        dm=np.sqrt(1/np.sum(weigth))
    else:
        mean=np.mean(data)
        dm=np.sqrt(1/len(data))*np.std(data)
    return mean,dm

def chi2(data,dy,model):
    return np.sum(((model - data)**2)/(dy**2))

def linear_regression_AB(x,y,w):
    w=w*np.ones(len(x))
    #pdb.set_trace()
    dw=np.sum(w) * \
            np.sum(w*(x**2)) - \
            (np.sum(w*x))**2
    A=( np.sum(w*(x**2))* \
        np.sum(w*y) - \
        np.sum(w*x) * \
        np.sum(w*x*y) ) / \
        dw
    B=( np.sum(w) * \
        np.sum(w*x*y) - \
        np.sum(w*y) * \
        np.sum(w*x) ) / \
        dw
    dA= np.sqrt(np.sum(((x)**2)*w) / dw)
    dB= np.sqrt(np.sum(w)/dw)
    return A,B,dA,dB

def cov(a,b,ddof=0):
    ma=np.mean(a)
    mb=np.mean(b)
    cov_ab=np.sum((a-ma) * (b-mb))/(len(a)-ddof)
    return cov_ab


class DataXY:
    def __init__(self,x,y,dx=0,dy=0,name="DataXY",x_label="x",y_label="y"):
        self.x=x
        self.y=y
        self.dx=dx*np.ones(len(x))
        self.dy=dy*np.ones(len(x))
        self.x_label=x_label
        self.y_label=y_label
        self.name=name

    def getLinearRegressionAB(self,w=None):
        if w is None:
            w=1/self.dy**2
        w=w*np.ones(len(self.x))
        dw=np.sum(w) * \
                np.sum(w*(self.x**2)) - \
                (np.sum(w*self.x))**2
        A=( np.sum(w*(self.x**2))* \
            np.sum(w*self.y) - \
            np.sum(w*self.x) * \
            np.sum(w*self.x*self.y) ) / \
            dw
        B=( np.sum(w) * \
            np.sum(w*self.x*self.y) - \
            np.sum(w*self.y) * \
            np.sum(w*self.x) ) / \
            dw
        dA= np.sqrt(np.sum(((self.x)**2)*w) / dw)
        dB= np.sqrt(np.sum(w)/dw)
        return A,B,dA,dB


    def getChi2(self):
        return np.sum(((self.getModel() - self.y)**2)/(self.dy**2))

    def getChi2Red(self,ddof=0):
        return self.getChi2()/(len(self.x)-ddof)

    def getModel(self,x=None):
        A,B,dA,dB = self.getLinearRegressionAB()
        if x is None:
            return A+B*self.x
        else:
            return A+B*x

    def getFitPlot(self,fmt="b.",fmt_m="r,-",x_lim=None):
        A,B,dA,dB = self.getLinearRegressionAB()

        fig,(ax_top,ax_bot) = plt.subplots(2,1, gridspec_kw={'height_ratios':[3,1]})
        fig.suptitle(self.name)

        ax_top.errorbar(self.x,self.y,xerr=self.dx,yerr=self.dy,fmt=fmt)
        if x_lim is None:
            x_lim=ax_top.get_xlim()
        ax_top.plot(x_lim,\
                     A+B*np.array(x_lim),\
                     fmt_m)
        ax_top.set_xlim(x_lim)
        ax_top.set_ylabel(self.y_label)
        ax_top.grid()
        ax_top.ticklabel_format(style="sci",scilimits=(0,0))

        res = self.y - self.getModel()
        dy_propagated = np.sqrt(self.dy**2 + (B*self.dx)**2)

        ax_bot.errorbar(self.x,res,yerr=dy_propagated,xerr=self.dx,fmt=fmt)
        ax_bot.axhline(0,color=fmt_m[0])
        ax_bot.set_xlim(x_lim)
        ax_bot.set_ylabel("Res "+self.y_label)
        ax_bot.set_xlabel(self.x_label)
        ax_bot.grid()


        return fig,(ax_top,ax_bot)
