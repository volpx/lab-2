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
    def __init__(self,x,y,dx=0,dy=0):
        self.x=x
        self.y=y
        self.dx=dx*np.ones(len(x))
        self.dy=dy*np.ones(len(x))
        
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
    
    def getModelAB(self):
        A,B,dA,dB = self.getLinearRegressionAB()
        return A*self.x+B
    
    def getChi2(self):
        return np.sum(((self.getModel() - self.y)**2)/(self.dy**2))
    
    def getFitPlot(self):
        A,B,dA,dB = self.getLinearRegressionAB()
        
        fig = plt.figure()
        
        ax_top = fig.add_subplot(2,1,1)
        
        DX = np.max(self.x)-np.min(self.x)
        x_mod = np.min(self.x) - .05*DX + 1.1 * DX * np.array([0,1]) # TODO: what?
        y_mod = B + A*x_mod
        ax_top.plot(x_mod,y_mod)
        ax_top.set_ylabel("dati")
        ax_top.grid()
        ax_top.set_xlim([np.min(self.x)-.06*DX,np.max(self.x)*.06*DX])
        
        res = self.y - (self.getModelAB())
        
        dy = np.sqrt(self.dy**2 + B*self.dx**2)
        
        ax_bot = fig.add_subplot(2,1,2)
        ax_bot.errorbar(self.x,res,self.dy)
        ax_bot.set_ylabel("residui")
        ax_bot.set_xlabel("x")
        ax_bot.grid()
        ax_bot.set_xlim([np.min(self.x)-.06*DX,np.max(self.x)+.06*DX])
        
        return fig,ax_top,ax_bot