#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fundamental Market Research (part 1)

Updated on Fri Feb 22 14:29:54 2019

"""
import numpy
import pandas
import math
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot 
import matplotlib.pyplot as plt
from linmodelclass import Linmodel
 
dataframe = pandas.read_excel('findata.xlsx')
data = dataframe.values
 
#Copied the data into a Linmodel object 
T = numpy.size(data[:, 0])
treasuryrate = Linmodel(data[1:, 0], T-1)
sp500real = Linmodel(data[:, 1], T)
usinflation = Linmodel(data[1:, 2], T-1)
peratio = Linmodel(data[:, 3], T)
sp500nominal = Linmodel(data[:, 4], T)
shiller = Linmodel(data[1:, 5], T-1)
sp500divyield = Linmodel(1/data[1:, 6], T-1)
sp500div = Linmodel(data[:, 7], T)
 
#shiller.Q_Q_plot() #shows
 

X = pandas.DataFrame({'shiller pe': shiller.data[:], 's&p div yield': sp500divyield.data[:], 'us infltn': usinflation.data[:] , 'treasuryrate': treasuryrate.data[:]})
ChangeNominal = [math.log(sp500nominal.data[i]) - math.log(sp500nominal.data[i+1]) for i in range(T-1)] 
ChangeReal = [math.log(sp500real.data[i]) - math.log(sp500real.data[i+1]) for i in range(T-1)]

#This code is adapated from https://datatofish.com/multiple-linear-regression-python/
def MultipleRegress(X,Y):
    X = sm.add_constant(X)
    Y_Multiple = sm.OLS(Y, X).fit()
    Y_Predictions = Y_Multiple.predict(X)
    print_Y_Multiple = Y_Multiple.summary()
    print(print_Y_Multiple)
    return Y_Multiple.params

#This finds the residual and prints them into a Q-Q plot
def residooMultipleRegression(Y, A, B1, B2, B3, B4, x1, x2, x3, x4): # Y is the ChangeNominal/Real, A is the intercept B1 - B4 are the corresponding coefficients of x1-x4
    residual = numpy.empty_like(Y)
    i = 0
    while i < T-1:
        PredicatedY = (B1 * x1[i]) + (B2 * x2[i]) + (B3 * x3[i]) + (B4 + x4[i])
        residual[i] = Y[i] - PredicatedY
        i += 1
    qqplot(residual, line = 's')
    plt.show()

Nominal_inter, Nominal_shil, Nominal_SPDivYield, Nominal_USinfl, Nominal_Treasury = MultipleRegress(X, ChangeNominal) #ChangeNominal multiple regression with Shiller, S&PDivYield, UsInflation, TreasuryRate

Real_inter, Real_shil, Real_SPDivYield, Real_USinfl, Real_Treasury = MultipleRegress(X, ChangeReal) #ChangeReal multiple regression with Shiller, S&PDivYield, UsInflation, TreasuryRate

residooMultipleRegression(ChangeNominal, Nominal_inter, Nominal_shil, Nominal_SPDivYield, Nominal_USinfl, Nominal_Treasury, shiller.data[:], sp500divyield.data[:], usinflation.data[:], treasuryrate.data[:] )
    
    
residooMultipleRegression(ChangeReal, Real_inter, Real_shil, Real_SPDivYield, Real_USinfl, Real_Treasury, shiller.data[:], sp500divyield.data[:], usinflation.data[:], treasuryrate.data[:] )



