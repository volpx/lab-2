#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:07:26 2018

@author: volpe
"""

## Library of functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat

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
    def __init__(self,
                 x,
                 y,
                 dx=0,
                 dy=0,
                 name="DataXY",
                 x_label="x",
                 y_label="y",
                 color="b"):
        self.x=np.array(x)
        self.y=np.array(y)
        self.dx=dx*np.ones(len(x))
        self.dy=dy*np.ones(len(x))
        self.x_label=x_label
        self.y_label=y_label
        self.name=name
        self.color=color

    @classmethod
    def from_csv_file(cls,filename,*args,**kwargs):
        # works for oscilloscope csv output
        df=pd.read_csv(filename,header=1)
        col=df.columns
        return cls(df[col[0]], df[col[1]], x_label=col[0], y_label=col[1],*args,**kwargs)

    def get_linear_regression_AB(self,w=None):
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


    def get_chi2(self,dy_prop=False):
        A,B,dA,dB = self.get_linear_regression_AB()
        if dy_prop:
            return np.sum(((self.get_model() - self.y)**2)/(self.dy**2 + (B*self.dx)**2))
        else:
            return np.sum(((self.get_model() - self.y)**2)/(self.dy**2))

    def get_chi2_red(self,ddof=0,dy_prop=False):
        return self.get_chi2(dy_prop=dy_prop)/(len(self.x)-ddof)

    def get_model(self,x=None):
        A,B,dA,dB = self.get_linear_regression_AB()
        if x is None:
            return A+B*self.x
        else:
            x=np.array(x)
            return A+B*x

    def get_fit_plot(self,
                   fmt="b.",
                   fmt_m="r,-",
                   x_lim=None,
                   save=False,
                   sci=True):
        A,B,dA,dB = self.get_linear_regression_AB()

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
        if sci:
            ax_top.ticklabel_format(style="sci",scilimits=(0,0))

        res = self.y - self.get_model()
        dy_propagated = np.sqrt(self.dy**2 + (B*self.dx)**2)

        ax_bot.errorbar(self.x,res,yerr=dy_propagated,xerr=self.dx,fmt=fmt)
        ax_bot.axhline(0,color=fmt_m[0])
        ax_bot.set_xlim(x_lim)
        ax_bot.set_ylabel("Res "+self.y_label)
        ax_bot.set_xlabel(self.x_label)
        ax_bot.grid()
        if sci:
            ax_bot.ticklabel_format(style="sci",scilimits=(0,0))
        if save:
            fig.savefig("data/"+self.name+".pdf",bbox_inches="tight")
        return fig,(ax_top,ax_bot)

    def pretty_print(self,ddof=0,dy_prop=False):
        A,B,dA,dB = self.get_linear_regression_AB()
        print("\nName =",self.name)
        print("y = A+Bx =",ufloat(A,dA),"+",ufloat(B,dB),"x")
        print("Chi2 =",self.get_chi2(dy_prop=dy_prop))
        print("Chi2Red =",self.get_chi2_red(ddof=ddof,dy_prop=dy_prop))

    @staticmethod
    def compare(datasets,title="Confronto datasets",sci=True,legend=True):
        fig=plt.figure()
        fig.suptitle(title)
        ax=fig.add_subplot(1,1,1)
        for i,data in enumerate(datasets):
            ax.errorbar(data.x, data.y, yerr=data.dy, xerr=data.dx, fmt=data.color+".")
            ax.plot([0, data.x.max()], data.get_model(x=[0, data.x.max()]), data.color+",-", label=data.name)
        ax.grid()
        ax.set_ylabel(datasets[0].y_label)
        ax.set_xlabel(datasets[0].x_label)
        if legend:
            ax.legend(loc="upper left",bbox_to_anchor=(-.01,1.01))
        if sci:
            ax.ticklabel_format(style="sci",scilimits=(0,0))
        if save:
            fig.savefig("data/"+title+".pdf",bbox_inches="tight")

class DataX:
    def __init__(self,x,dx=0,name="DataX",x_label="x"):
        self.x=np.array(x)
        self.dx=dx*np.ones(len(x))
        self.x_label=x_label
        self.name=name
        self.color=color

    def get_mean(self,weigth=None):
        if weigth is None:
            mean=np.mean(self.x)
            dm=np.sqrt(1/len(data))*np.std(data)
        else:
            if weigth is True:
                weigth=self.dx**(-2)
                print("Weighted!!!!")
            weigth=weigth*np.ones(len(self.x))
            mean=np.sum(self.x*weigth)/np.sum(weigth)
            dm=np.sqrt(1/np.sum(weigth))
        return mean,dm

    def get_chi2(self):
        M,_ = self.get_mean(weigth=True)
        return np.sum(((M - self.x)**2)/(self.dx**2))

    def get_chi2_red(self,ddof=0):
        return self.get_chi2()/(len(self.x)-ddof)

    def get_fit_plot(self,
                   save=False,
                   sci=True,
                   order=None,
                   colors=None,
                   names=None,
                   k=1,
                   mean=True,
                   add_values=None):
        if colors is None:
            colors=plt.cm.Set3(np.linspace(0,1,len(self.x)+len(add_values)))
        if order is None:
            order=list(range(len(self.x)))
        M,dM = self.get_mean(weigth=True)
        height=1/(len(colors)+(1 if mean else 0))

        fig = plt.figure()
        fig.suptitle(self.name)

        ax=fig.add_subplot(1,1,1)

        # need j for changing colors even after the first loop
        j=0
        for j,i in enumerate(order):
            if names is None:
                ax.axvline(x=self.x[i],color=colors[j])
            else:
                ax.axvline(x=self.x[i], color=colors[j], name=names[i])
            if self.dx[0] != 0:
                ax.axvspan(xmin=self.x[i]-k*self.dx[i], xmax=self.x[i]+k*self.dx[i], ymin=1-height*(j+1), ymax=1-height*j, color=colors[j], alpha=0.2)

        for value in add_values:
            ax.axvline(x=value[0], color=colors[j], label=names[j])
            ax.axvspan(xmin=value[0]-k*value[1], xmax=value[0]+k*value[1], ymin=1-height*(j+1), ymax=1-height*j, color=colors[j], alpha=0.2)
            j+=1

        ax.set_xlabel(self.x_label)
        ax.set_yticks([])
        ax.legend(loc="center left",bbox_to_anchor=(1.03,.5))
        if save:
            fig.savefig("data/"+self.name+".pdf",bbox_inches="tight")
        return fig, ax

    def pretty_print(self, ddof=0, dy_prop=False):
        Mw,dMw=self.get_mean(weigth=True)
        M,dM=self.get_mean()
        print("\nName =",self.name)
        print("Mean x =",ufloat(M,dM))
        print("WMean x =",ufloat(Mw,dMw))
        print("Chi2 =",self.get_chi2())
        print("Chi2Red =",self.get_chi2_red(ddof=ddof))
