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

for n in ["","no"]:
    for s in range(1,5+1):
        for t in range(0,6):
            print("File:","{n} cap, serie {s}, try {t}".format(n=n,s=s,t=t))
            x=DataXY.from_csv_file_special1(scope_folder+fn.format(n=n,s=s,t=t),
                                            name="{n} cap, serie {s}, try {t}".format(n=n,s=s,t=t),
                                            color="b",
                                            y_col="2",
                                            dx=0,dy=0).get_plot(save=True,out_folder=out_folder)
