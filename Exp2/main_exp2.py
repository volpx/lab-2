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
from functions import DataXY

fn="sc_{n}cap_r{s}c{t}.csv"
scope_folder="data/raw_data_from_scope/"
out_folder="data/"

rs_n=np.array([1e3,1e4,1e5,3e4,3.73e5])
rs_dmm=np.array([998.2,9.917e3,99.66e3,29.82e3,373.2e3])
i_min_a=np.array(
      [[[2191, 2192, 2190, 2188, 2191, 2191],
        [2238, 2239, 2238, 2237, 2239, 2240],
        [1337, 1337, 1337, 1337, 1338, 1338],
        [1041, 1043, 1043, 1043, 1043, 1043],
        [ 414,  414,  414,  414,  414,  414]],

       [[  83,   83,   83,   83,   81,   79],
        [1216, 1222, 1208, 1222, 1223, 1224],
        [ 944,  943,  943,  943,  943,  943],
        [1013, 1018, 1020, 1017, 1015, 1018],
        [1746, 1746, 1746, 1745, 1745, 1746]]])
i_max_a=np.array( #2.5 tau_n
      [[[14669, 14670, 14668, 14666, 14669, 14669],
        [14635, 14636, 14635, 14634, 14636, 14637],
        [13795, 13795, 13795, 13795, 13796, 13796],
        [15952, 15954, 15954, 15954, 15954, 15954],
        [16501, 16501, 16501, 16501, 16501, 16501]],

       [[  962,   962,   962,   962,   960,   958],
        [ 9943,  9949,  9935,  9949,  9950,  9951],
        [ 8775,  8774,  8774,  8774,  8774,  8774],
        [ 7742,  7747,  7749,  7746,  7744,  7747],
        [11130, 11130, 11130, 11129, 11129, 11130]]])
cap_n=50e-9
nocap_n=176e-12

for ni,n in enumerate(["","no"]):
    pass
    for s in range(1,5+1):
        for t in range(0,6):
            print("File:","{n} cap, serie {s}, try {t}".format(n=n,s=s,t=t))

            df=pd.read_csv(scope_folder+fn.format(n=n,s=s,t=t),header=0,skiprows=[1])

            tau_n = (cap_n if ni==0 else nocap_n) * rs_dmm[s-1]
            i_min=i_min_a[ni][s-1][t]
#            i_max=i_max_a[ni][s-1][t]
            tm_off=df["x-axis"][i_min]

            #find the start of discharging index
#            tresh=4
#            i=0
#            while ( df["1"][i]>tresh or np.isnan(df["1"][i]) ):
#                i+=1
#            i_min=i
            #find the 2.5 tau index
            while (df["x-axis"][i]<tm_off+2*tau_n):
                i+=1
            i_max=i


            dat=DataXY.from_csv_file_special2(scope_folder+fn.format(n=n,s=s,t=t),
                                            name="{n} cap, serie {s}, try {t}".format(n=n,s=s,t=t),
                                            color="b",
                                            y_col="2",
                                            i_min=i_min,
                                            i_max=i_max,
                                            dx=0,dy=0)
            dat.x=dat.x-tm_off
            dat.y=np.log(np.abs(dat.y))
            F=np.vstack([ np.ones(dat.x.size), dat.x, 1/dat.y]).T
            lam=dat.get_general_regression(F,dy=1)
            pass

