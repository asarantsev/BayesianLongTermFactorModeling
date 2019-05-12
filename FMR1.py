#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fundamental Market Research (part 1)

Updated on Fri Feb 22 14:29:54 2019

"""
import numpy
 
import pandas
 
dataframe = pandas.read_excel('findata.xlsx')
data = dataframe.values
 
import scipy
from scipy import stats
 
rate = data[:, 0]
sp500real = data[:, 1]
usinflation = data[:, 2]
peratio = data[:, 3]
sp500nominal = data[:, 4]
shiller = data[:, 5]
sp500divyield = 1/data[:, 6] #The inverse of the dividend yield
sp500div = data[:, 7]
T = numpy.size(rate)
 
def ar(x):
    return stats.linregress(x[1:], x[:T-1])
print(ar(sp500divyield))
print(ar(peratio))
print(ar(usinflation))
print(ar(shiller))
 
#this is Chyna Metz-Bannister's work done on February 18, 2019
#slightly modifed by Lissa Callahan February 22, 2019