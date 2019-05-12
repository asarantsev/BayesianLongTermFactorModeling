# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
sp500divyield = data[:, 6]
sp500div = data[:, 7]
cpi = data[:, 8]
T = numpy.size(rate)

def ar(x):
    return stats.linregress(x[1:], x[:numpy.size(x)-1])
#print(ar(sp500divyield))
#print(ar(peratio))
#print(ar(usinflation))
#print(ar(shiller))

tenyearinflation = []

for t in range(1641):
    i10 = (1/10)*((cpi[t]/cpi[t+120])-1)
    tenyearinflation.append(i10)

print(ar(tenyearinflation))    
print(stats.linregress(tenyearinflation[12:1640], usinflation[:1628]))
    
earningsyield = 1/peratio

tenyeardivyield = []

for t in range(1641):
    current10dy = numpy.mean([sp500div[t+12*k] for k in range(10)])/sp500real[t]
    tenyeardivyield.append(current10dy)
    
print(ar(tenyeardivyield))
print(stats.linregress(tenyeardivyield[12:1640], sp500divyield[:1628]))

    

#this is Chyna Metz-Bannister's work done on March 26, 2019