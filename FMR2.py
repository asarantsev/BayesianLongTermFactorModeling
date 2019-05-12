#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:37:33 2019

FMR Step 2: Perform Autoregression of order q
Refer to ARMA_parameters folder for output

@author: lissacallahan
"""

import os # operating system
import pandas as pd
from statsmodels.tsa.arima_model import ARMA
#Autoregressive Moving Average ARMA(p,q) Model

dataframe = pd.read_excel('~/Dropbox/Fundamental Market Research/findata.xlsx')
data = dataframe.values


usinflation = data[:, 2]
peratio = data[:, 3]
shiller = data[:, 5]
pdratio = 1/data[:, 6] #The inverse of the dividend yield


def run_arma (values, order=(1,0)):
    mod = ARMA(values, order=order)
    return mod.fit(disp=0) # suppresses fit data

# This will print all 100 iterations of the summary table, each to a separate file
# Code for showing just one at a time shown below
def print_summary(data, label):
    for p in range(1,6):
        for q in range(0,5):
            print(f"{label}  p: {p}  q: {q}")
            filepath = os.path.expanduser(f"~/downloads/arma/{label}_{p}_{q}.txt")
            file = open(filepath,"w") 
            try:
                #AR order p, MA order q: AR model only if q=0
                results = (run_arma(data, (p,q)).summary().as_text())
                file.write(results)  
            except Exception as err:
                file.write(str(err))
            file.close() 
        
# ARMA Model Results Summary, print to file      
# each will create 25 .txt files for respective data
print_summary(pdratio, 'pdratio') 
print_summary(peratio, 'peratio') 
print_summary(shiller, 'shiller') 
print_summary(usinflation, 'usinflation') 

"""
# To show just one summary without printing to a file:
# where "data" is appropriate findata column, "p" is AR order and "q" is MA order
#.params prints just the coefficients as an array (doesn't show std err)
print(run_arma(data, (p,q)).params)
#.summary() shows a table with all applicable information, later used in Excel spreadsheets
print(run_arma(data, (p,q)).summary())
# for example print(run_arma(pdratio, (1,1)).summary())
"""
