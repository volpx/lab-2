#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:45:48 2018

@author: volpe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from uncertainties import ufloat
from functions import *

# %%
C=10e-9 # F
L=1e-6 # H

w=np.exp(np.linspace(np.log(10),np.log(1e8),1000))
s=1j*w

H=s**2*L*C/(1+s**2*L*C)

# %%
fig_bode,(_,_) = bode_plot(x=w, H=H,title='RL filter')
plt.show()
