#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:45:48 2018

@author: volpe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from functions import DataXY,general_regression,bode_plot

w=2*np.pi*10**np.linspace(0,10,30)
s=1j*w

C=10e-9
R=1e3

ZC=1/s/C
ZR=R*s**0

H=ZC/(ZR+ZC)
w3db=1/C/R

fig,(ax_mod,ax_ph)=bode_plot(w,H,fmt='b,-')
ax_mod.axvline(w3db)
ax_ph.axvline(w3db)
ax_mod.axhline(-3)
ax_ph.axhline(0)
plt.show()