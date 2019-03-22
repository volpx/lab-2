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

def chi2red(data,dy,model=0,ddof=0):
    return np.sum(((model - data)**2)/(dy**2)) / (data.size - ddof)

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

def general_regression(F,y,dy=None):
    """
        F : coefficients NxM array, the columns are the f(x) or f(y), the rows are the different points
    """
    # check input
    if not isinstance(F,np.ndarray):
        print('F not correct')
    if not isinstance(y,np.ndarray):
        print('y not correct')
    err_given=True
    if dy is None:
        dy=1
        err_given=False
    dy = dy * np.ones(y.size)

    # N=number of points
    N=F.shape[0]
    # M=number of functions
    M=F.shape[1]

    # array of... data (?)
    V=np.empty( (M,) )
    # correlation smth (?)
    G=np.empty( (M,M) )

    # calculating correlation matrix
    for i in range(M):
        # each element is the sum(f(x,y)*y/dy**2) (?)
        V[i] = np.sum( F[:,i] * y / (dy**2))
        for j in range(M):
            G[i,j] = np.sum( F[:,i] * F[:,j] / (dy**2))

    # C is probably the covariance matrix
    C = np.linalg.inv(G)

    # get params
    lam = C @ V
    # C is square
    dlam = np.sqrt((np.eye(C.shape[0])*C) @ np.ones(C.shape[0]))
    # y of the model
    y_fit = F @ lam
    # residuals
    y_res = y - y_fit

    dof = N - lam.size

    # calculate mean of residuals
    y_res_m = np.sum(y_res) / ( dof )
    y_res_rms=np.sqrt(np.sum( y_res**2 )/dof)

    # calculate chi2red
    chi2red=np.sum(y_res ** 2 / dy**2) / dof

    if not err_given:
        C *= chi2red
        dlam = np.sqrt((np.eye(C.shape[0])*C) @ np.ones(C.shape[0]))

    return lam,dlam,C,chi2red,dof,y_res_m,y_res_rms

def bode_plot(x, H, dB=True,
              xerr=None,Herr=None,err=False,
              x_label='ω [rad * s^-1]',y_label='H',
              y_lim_mod=None,fmt='b.',title=None,
              ext_fig=None):
    """
        x: (N,) the x coordinates, usually rad/s
        H: (N,2) matrix with module and phase of H function for each x
            (N,) vector of complex numbers of H function
        err: plot with errors switch
        xerr: (N,) error on x
        Herr: (2,N) error on H for mod and phase
        y_lim_mod: (2,) set limits on module plot
        fmt: format string for the plot
        ext_fig: external figure that has exactly 2 axes in which the plot will be made,
                    if passed no labels or title will be changed
    """

    if H.ndim is 2:
        H_mod=H[:,0]
        H_phase=H[:,1]
    elif H.ndim is 1:
        #each element of H is a complex number
        H_mod=np.abs(H[:])
        H_phase=np.angle(H[:])

    if ext_fig is None:
        fig=plt.figure()

        if title is not None:
            fig.suptitle(title)

                        #[left,bot,width,height]
        ax_top=fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
        ax_bot=fig.add_axes([0.1, 0.1, 0.8, 0.4])

        ax_top.set_xscale('log')
        ax_bot.set_xscale('log')

        y_label='|'+y_label+'|' + ('dB' if dB else '')
        ax_top.set_ylabel(y_label)
        ax_bot.set_ylabel('φ [rad]')
        ax_bot.set_xlabel(x_label)

        ax_bot.grid()
        ax_top.grid()
    else:
        fig=ext_fig

        ax_top,ax_bot=fig.get_axes()

    if dB:
        if err:
            mod_err=20/np.log(10)*Herr[0]/H_mod
        H_mod=20*np.log10(H_mod)
    else:
        if err:
            mod_err=Herr[0]

    if err:
        ax_top.errorbar(x,H_mod,xerr=xerr,yerr=mod_err,fmt=fmt)
        ax_bot.errorbar(x,H_phase,xerr=xerr,yerr=Herr[1],fmt=fmt)
    else:
        ax_top.plot(x,H_mod,fmt)
        ax_bot.plot(x,H_phase,fmt)

    return fig,(ax_top,ax_bot)

def fit_plot(x,y,model,
             yerr=0,xerr=0,
             title=None,
             x_label=None,
             y_label=None,
             fmt="b.",
             fmt_m="r,-",
             x_lim=None,
             save=False,
             sci=True):


    fig,(ax_top,ax_bot) = plt.subplots(2,1, gridspec_kw={'height_ratios':[3,1]})
    if title is not None:
        fig.suptitle(title)

    ax_top.errorbar(x,y,xerr=xerr,yerr=yerr,fmt=fmt)
    if x_lim is None:
        x_lim=ax_top.get_xlim()
    ax_top.plot(x, model, fmt_m)
    ax_top.set_xlim(x_lim)
    if y_label is not None:
        ax_top.set_ylabel(y_label)
    ax_top.grid()
    if sci:
        ax_top.ticklabel_format(style="sci",scilimits=(0,0))

    # res plot
    res = y - model

    # maybe dy propagated

    ax_bot.errorbar(x,res,yerr=yerr,xerr=xerr,fmt=fmt)
    ax_bot.axhline(0,color=fmt_m[0])
    ax_bot.set_xlim(x_lim)
    if y_label is not None:
        ax_bot.set_ylabel(y_label+' res')
    if x_label is not None:
        ax_bot.set_xlabel(x_label)
    ax_bot.grid()
    if sci:
        ax_bot.ticklabel_format(style="sci",scilimits=(0,0))

    if save:
        fig.savefig(save,bbox_inches="tight")
    return fig,(ax_top,ax_bot)

def par(z1,z2):
    """
        circuit parallel
    """
    return 1/(1/z1+1/z2)

def find_divisions(file):
    '''
        for oscilloscope use
        returns [v_div,t_div] the v_div is of the first channel in the file
    '''
    #find deltas
    with open(file,'r') as f:
        words=f.read().split()
        i=0

        # find the v scale
        while words[i] != 'Scale':
            i+=1
        i+=1
        dv_s=words[i][:-2]

        # find the t scale
        while words[i] != 'HORIZONTAL':
            i+=1
        while words[i] != 'Scale':
            i+=1
        i+=1
        dt_s=words[i][:-2]

        #adjust the measurement unit
        dv=float(dv_s[:-2])
        v_mu=dv_s[-2:]
        if v_mu[0] is 'm':
            dv*=1e-3
        elif v_mu[0] is 'u':
            dv*=1e-6
        elif v_mu[0] is 'n':
            dv*=1e-9

        dt=float(dt_s[:-2])
        t_mu=dt_s[-2:]
        if t_mu[0] is 'm':
            dt*=1e-3
        elif t_mu[0] is 'u':
            dt*=1e-6
        elif t_mu[0] is 'n':
            dt*=1e-9

        return dv,dt
def fit_sine_poly(t,x,polyord,freqs,err=None,t0=0,center=False,tran=None):
    """
        LSQ routine that fits data x(t) to a polynomial (order polyord) and any
        number of cos / sin terms with frequencies listed in freqs

        INPUT:
            t: input t
            x: input x
            polyord: order of the polinomial
            freqs: list of frequancies

        OUTPUT:
            fitout: [a0, a1, ..., a_polyord, C1, S1, C2, S2, ...]
            dfit_out: literally fitout with a d before
            C: covariance matrix
            chi2: chi2 when errors are given
            N_DOF: number of degrees of freedom
    """

    if tran is not None:
        inds=(tran[0]<=t) & (t<=tran[1])
        tin=t[inds]
        xin=x[inds]
    else:
        tin=t
        xin=x

    if err is None:
        err=1

    if center:
        t_c = np.mean(tin)
    else:
        t_c=0

    F = np.empty((xin.size,polyord))

    for m in range(polyord):
        f = (tin-t_c)**m
        F[:,m]=f

    freqs=np.array(freqs)
    Ftmp=np.empty((xin.size,2*freqs.size))
    if freqs.size > 0:
        for m in range(freqs.size):
            Ftmp[:,2*m] = np.cos(2*np.pi*freqs[m]*(tin-t0))
            Ftmp[:,2*m+1] = np.sin(2*np.pi*freqs[m]*(tin-t0))

    F=np.append(F,Ftmp,axis=1)
    M = F.shape[1]
    G = np.empty((M,M))
    V = np.empty((M,))
    for i in range(M):
        V[i] = np.sum(F[:,i] * xin / err**2)
        for j in range(M):
            G[i,j] = np.sum(F[:,i] * F[:,j] / err**2)

    C=np.inv(G)
    fit_out = C * V

    x_fit = F*fit_out
    dx_res = xin -x_fit
    N_DOF=xin.size - fit_out.size
    dx_mean = np.sum(dx_res) / N_DOF

    chi2=np.sum(dx_res**2/err**2)/N_DOF

    dfit_out = np.sqrt(np.diag(C))

    sigma = np.sqrt(np.sum(dx_res**2)/N_DOF)

######################### SHIT PAST THIS #######################################
class DataXY:
    def __init__(self,
                 x,
                 y,
                 dx=1,
                 dy=1,
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
    def from_csv_file(cls,filename,x_col=0,y_col=1,*args,**kwargs):
        # works for oscilloscope csv output
        df=pd.read_csv(filename,header=1)
        col=df.columns
        return cls(df[col[x_col]], df[col[y_col]], x_label=col[x_col], y_label=col[y_col],*args,**kwargs)

    @classmethod
    def from_csv_file_special0(cls,
                               filename,
                               x_col=0,
                               y_col=1,
                               y_min=-np.inf,
                               y_max=np.inf,
                               middle=0.5,
                               *args,**kwargs):
        # works for oscilloscope csv output
        df=pd.read_csv(filename,header=1)
        col=df.columns

        #start in the middle and find min and max index
        l=df[col[x_col]].size

        i_min=int(middle*l)
        while (i_min > 0 and (y_min < df[col[y_col]][i_min] < y_max) ):
            i_min-=1
        i_max=l//2
        while (i_max < l and (y_min < df[col[y_col]][i_max] < y_max) ):
            i_max+=1

        return cls(df[col[x_col]][i_min:i_max].values,
                   df[col[y_col]][i_min:i_max].values,
                   x_label=col[x_col], y_label=col[y_col],*args,**kwargs)

    @classmethod
    def from_csv_file_special1(cls,
                               filename,
                               x_col=0,
                               y_col=1,
                               y_min=-np.inf,
                               y_max=np.inf,
                               middle=0.5,
                               *args, **kwargs):
        # works for oscilloscope csv output
        df=pd.read_csv(filename,header=0,skiprows=[1])
        col=df.columns
        if isinstance(y_col,int):
            y_col=col[y_col]
        if isinstance(x_col,int):
            x_col=col[x_col]

        #start in the middle and find min and max index
        l=df[x_col].size

        i_min=int(middle*l)
        while (i_min > 0 and (y_min < df[y_col][i_min] < y_max) ):
            i_min-=1
        i_max=l//2
        while (i_max < l and (y_min < df[y_col][i_max] < y_max) ):
            i_max+=1

        return cls(df[x_col][i_min:i_max].values,
                   df[y_col][i_min:i_max].values,
                   x_label=x_col, y_label=y_col,*args,**kwargs)

    @classmethod
    def from_csv_file_special2(cls,
                               filename,
                               x_col=0,
                               y_col=1,
                               i_min=0,
                               i_max=None,
                               middle=0.5,
                               *args, **kwargs):
        # works for oscilloscope csv output
        df=pd.read_csv(filename,header=0,skiprows=[1])
        col=df.columns
        if isinstance(y_col,int):
            y_col=col[y_col]
        if isinstance(x_col,int):
            x_col=col[x_col]

        #start in the middle and find min and max index
        l=df[x_col].size
        if i_max is None:
            i_max=l

        return cls(df[x_col][i_min:i_max].values,
                   df[y_col][i_min:i_max].values,
                   x_label=x_col, y_label=y_col,*args,**kwargs)


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

    def get_gen_reg(
            self,
            F,
            plot_save=False,
            fmt="b,",
            fmt_m="r,",
            pp=False):

        lam,dlam,C,chi2red,dof,y_res_m,y_res_rms = general_regression(F,self.y,self.dy)

        if pp:
            print('Fit results:')
            print('  chi2red:',chi2red,'@ dof:',dof)
            print('  y_res_m:',y_res_m)
            print('  y_res_rms:',y_res_rms)
            print('  lam:')
            for i in range(lam.size):
                print('    {i})'.format(i=i),ufloat(lam[i],dlam[i]))

        if plot_save:

            fig,(ax_top,ax_bot) = plt.subplots(2,1, gridspec_kw={'height_ratios':[3,1]})
            fig.suptitle(self.name)


            #mod
#            ax_top.set_yscale('log')
            #ax_top.set_xscale('log')
#            ax_bot.set_yscale('log')
            #ax_bot.set_xscale('log')


            ax_top.errorbar(self.x,self.y,xerr=self.dx,yerr=self.dy,fmt=fmt)
#            ax_top.errorbar(self.x,self.y,xerr=0,yerr=0,fmt=fmt)
            x_lim=ax_top.get_xlim()
            ax_top.plot(self.x, y_fit, fmt_m)
            ax_top.set_xlim(x_lim)
            ax_top.set_ylabel(self.y_label)
            ax_top.grid()
#            ax_top.ticklabel_format(style="sci",scilimits=(0,0))

            ax_bot.errorbar(self.x,y_res,yerr=dy,xerr=self.dx,fmt=fmt)
#            ax_bot.errorbar(self.x,y_res,yerr=0,xerr=0,fmt=fmt)
            ax_bot.axhline(0,color=fmt_m[0])
            ax_bot.set_xlim(x_lim)
            ax_bot.set_ylabel("Res "+self.y_label)
            ax_bot.set_xlabel(self.x_label)
            ax_bot.grid()
#            ax_bot.ticklabel_format(style="sci",scilimits=(0,0))
            fig.savefig("data/"+self.name+".pdf",bbox_inches="tight")

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

    def get_plot(self,
                   fmt="b.",
                   x_lim=None,
                   save=False,
                   sci=True,
                   err=False,
                   out_folder="data/"):

        fig = plt.figure()
        ax=fig.add_subplot(1,1,1)
        fig.suptitle(self.name)

        if err:
            ax.errorbar(self.x,self.y,xerr=self.dx,yerr=self.dy,fmt=fmt)
        else:
            ax.plot(self.x,self.y,fmt)

        ax.set_ylabel(self.y_label)
        ax.set_xlabel(self.x_label)
        ax.grid()
        if sci:
            ax.ticklabel_format(style="sci",scilimits=(0,0))

        if save:
            fig.savefig(out_folder+self.name+".pdf",bbox_inches="tight")
        return fig, ax

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
    def __init__(self,x,dx=1,name="DataX",x_label="x"):
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

class DataSetXY:
    def __init__(self,name="DataSetXY",data=[]):
        self.name=name
        self.data=data

    @classmethod
    def from_csv_files(cls,filenames,*args,**kwargs):
        data=[]
        for fn in filenames:
            data.append(DataXY.from_csv_file(fn,*args,**kwargs))
