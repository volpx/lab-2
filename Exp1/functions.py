#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:07:26 2018

@author: volpe
"""

## Library of functions
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat


def w_mean(data,weigth=None):
    """
        Does the weighted mean of values in array data, if no weights given do arithmetic mean
    """
    if weigth.any():
        mean=np.sum(data*weigth)/np.sum(weigth)
        dm=np.sqrt(1/np.sum(weigth))
    else:
        mean=np.mean(data)
        # dm is standard deviation divided by the square root of the number of data given
        dm=np.sqrt(1/len(data))*np.std(data)
    return mean,dm

def chi2(data,dy,model):
    """
        calculate the chi^2 given array data, int or array dy and array model
    """
    return np.sum(((model - data)**2)/(dy**2))

def linear_regression_AB(x,y,w):
    """
        calculate the linear regression A+B*x given data x, data y and weights w
    """
    #make sure w in an array
    w=w*np.ones(len(x))
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
    # return a tuple
    return A,B,dA,dB

def cov(a,b,ddof=0):
    ma=np.mean(a)
    mb=np.mean(b)
    cov_ab=np.sum((a-ma) * (b-mb))/(len(a)-ddof)
    return cov_ab


class DataXY:
    """
        My class DataXY represent a elementary succession of x,y couple of data with the operations you could do on them
    """
    def __init__(self,x,y,dx=0,dy=0,name="DataXY",x_label="x",y_label="y"):
        self.x=np.array(x)
        self.y=np.array(y)
        #optional value dx and dy is 0 if not given
        self.dx=dx*np.ones(len(x))
        self.dy=dy*np.ones(len(x))
        # save optional x and y columns names
        self.x_label=x_label
        self.y_label=y_label
        # save a name for the frame
        self.name=name

    def getLinearRegressionAB(self,w=None):
        """
            do the normal linear regression on the class with w=1/dy^2 if not given
        """
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


    def getChi2(self,dy_prop=False):
        """
            get a chi2 value optionally propagating uncertaintes after linear regression
        """
        A,B,dA,dB = self.getLinearRegressionAB()
        if dy_prop:
            return np.sum(((self.getModel() - self.y)**2)/(self.dy**2 + (B*self.dx)**2))
        else:
            return np.sum(((self.getModel() - self.y)**2)/(self.dy**2))

    def getChi2Red(self,ddof=0,dy_prop=False):
        """
            get the reduced chi2 given the delta degrees of freedom ddof
        """
        return self.getChi2(dy_prop=dy_prop)/(len(self.x)-ddof)

    def getModel(self,x=None):
        """
            get y values of fit model for each self.x given or only for the passed x
        """
        A,B,dA,dB = self.getLinearRegressionAB()
        if x is None:
            return A+B*self.x
        else:
            return A+B*x

    def getFitPlot(self,fmt="b.",fmt_m="r,-",x_lim=None,save=False):
        """
            make a summary plot with fit an residues
        """
        A,B,dA,dB = self.getLinearRegressionAB()

        # make a figure already with two plots taking 75% and 25% of vertical space respectively
        fig,(ax_top,ax_bot) = plt.subplots(2,1, gridspec_kw={'height_ratios':[3,1]})
        # set title
        fig.suptitle(self.name)
        
        # plot my data
        ax_top.errorbar(self.x,self.y,xerr=self.dx,yerr=self.dy,fmt=fmt)
        # take the current x limits of the plot area for later if not passed
        if x_lim is None:
            x_lim=ax_top.get_xlim()
        # plot the fit from left limit to right limit
        ax_top.plot(x_lim,\
                     A+B*np.array(x_lim),\
                     fmt_m)
        # re-set the x limits
        ax_top.set_xlim(x_lim)
        # set y label
        ax_top.set_ylabel(self.y_label)
        # put grid
        ax_top.grid()
        # set ticks with scientific notation
        ax_top.ticklabel_format(style="sci",scilimits=(0,0))
        
        #get residues
        res = self.y - self.getModel()
        # get propagated uncertainty
        dy_propagated = np.sqrt(self.dy**2 + (B*self.dx)**2)
        
        #plot residues with their uncertainties
        ax_bot.errorbar(self.x,res,yerr=dy_propagated,xerr=self.dx,fmt=fmt)
        # plot an horizontal line at y=0 representing the model here
        ax_bot.axhline(0,color=fmt_m[0])
        ax_bot.set_xlim(x_lim)
        ax_bot.set_ylabel("Res "+self.y_label)
        ax_bot.set_xlabel(self.x_label)
        ax_bot.grid()
        ax_bot.ticklabel_format(style="sci",scilimits=(0,0))
        # return the figure and its axes
        return fig,(ax_top,ax_bot)

    def prettyPrint(self,ddof=0,dy_prop=False):
        """
            decently print a DataXY summary
        """
        A,B,dA,dB = self.getLinearRegressionAB()
        print("\nName =",self.name)
        print("y = A+Bx =",ufloat(A,dA),"+",ufloat(B,dB),"x")
        print("Chi2 =",self.getChi2(dy_prop=dy_prop))
        print("Chi2Red =",self.getChi2Red(ddof=ddof,dy_prop=dy_prop))
