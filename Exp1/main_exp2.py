#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:57:32 2018

@author: volpe
"""

import pandas as pd
from functions import DataXY
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

from functions import DataXY
#
data=[pd.read_csv("data/AmpMon10V5mA.csv"),
         pd.read_csv("data/AmpMon2V0.5mA.csv"),
         pd.read_csv("data/AmpVal10V5mA.csv"),
         pd.read_csv("data/AmpVal2V0.5mA.csv")]

for df in data:
    df["mA"]*=1e-3
    df.columns=["V","A"]
data_fs=np.array([[10,5e-3],[2,500e-6],[10,5e-3],[2,500e-6]])
data_Delta=np.array([[0.2,100e-6],[0.04,10e-6],[0.2,100e-6],[0.04,10e-6]])
data_sigma=data_Delta/np.sqrt(12)

dataseries=[DataXY(data[0]["V"],data[0]["A"],\
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

dataseries[0].getFitPlot()[0].savefig("data/AmpMon10V5mA.svg")
dataseries[1].getFitPlot()[0].savefig("data/AmpMon2V0.5mA.svg")
dataseries[2].getFitPlot()[0].savefig("data/AmpVal10V5mA.svg")
dataseries[3].getFitPlot()[0].savefig("data/AmpVal2V0.5mA.svg")

if __name__ == "__main__":
    main()
