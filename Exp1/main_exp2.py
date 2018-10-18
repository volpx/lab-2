#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:57:32 2018

@author: volpe
"""
r
from functions import DataXY
import numpy as np
import matplotlib.pyplot as plt

def main():
    x=np.arange(100)
    y=5+3*x+np.random.rand(len(x))
    data=DataXY(x,y,1,1)
    fig,ax1,ax2=data.getFitPlot()
    plt.show()

if __name__ == "__main__":
    main()
