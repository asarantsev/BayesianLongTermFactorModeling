#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:16:41 2019

**For each autoregression from Step 2, test normality of residuals via Q-Q plots**

@author: lissacallahan
"""
import pandas as pd
import pylab
import numpy
from matplotlib import pyplot
from scipy.stats.distributions import norm
from statsmodels.graphics.gofplots import qqplot
#instead of statsmodels.graphics.gofplots import qqplot


dataframe = pd.read_excel('~/Dropbox/Fundamental Market Research/findata.xlsx')
data = dataframe.values

I = data[:, 2]
P = data[:, 3]
S = data[:, 5]
D = 1/data[:, 6] #The inverse of the dividend yield

#The equation needed (order 2 example), where t = q, ..., T, T =1762, q = 2
#EpAR2 = P[t] - AR2b_p - ((AR2a_p1)*(P[t-1])) - ((AR2a_p2)*(P[t-2]))


#For PE Ratio data set "P"
aparms = {
  "2": [1.5509	, -0.5843], # dict to hold AR order 2 alpha values
  "3": [1.4714	, -0.3734 , -0.1358] # dict to hold AR order 3 alpha values
}

#For PD Ratio data set "D"
aparms = {
  "2": [1.2385 , -0.2421], # dict to hold AR order 2 alpha values
  "3": [1.2494 , -0.2981	 , 0.0453] # dict to hold AR order 3 alpha values
}

#For Shiller data set "S"
aparms = {
  "2": [1.2408 , -0.2464], # dict to hold AR order 2 alpha values
  "3": [1.2520 , -0.3028 , 0.0455] # dict to hold AR order 3 alpha values
}

#For US Inflation data set "I"
aparms = {
  "2": [1.2083 , -0.2454], # dict to hold AR order 2 alpha values
  "3": [1.1846 , -0.1286	 , -0.0966] # dict to hold AR order 3 alpha values
}

def calc_residual(data, yint, order):
    result = [] #initialize empty array
    #print(data[order:]) 
    for idx in range(order, len(data)): # t= q, ... , T, from order 2 or 3 to end of array
        result.append(data[idx] - yint + alpha(idx, order, data)) #The equation solved for E resisdual
        # call alpha function to determine last terms depending on order
    return result

def alpha(index, order, data):
    result = 0 # initialize #idx is 0 for first a, idx is 1 for 2nd a
    for idx in range(0, order):
        # iterating over row (key) and then element in dict array (values)
        a = aparms[str(order)][idx] #idx is 0 for first a, idx is 1 for 2nd a
        result -= a * data[index - 1 - idx]
    return result


def save_result(vals, label):
    with open(f'/Users/lissacallahan/downloads/{label}result.csv', 'w') as f:
        for item in vals:
            f.write("%s\n" % item)

#Label name: P2 = PE Ratio order 2            
P2 = calc_residual(P, 15.7675, 2) # data, beta value, order 2             
save_result(P2, 'P2') # run code and save to a file
P3 = calc_residual(P, 15.7621, 3) # data, beta value, order 3
save_result(P3, 'P3')
            
#Label name: D2 = PD Ratio order 2 
D2 = calc_residual(D, 29.6587, 2) # data, beta value, order 2             
save_result(D2, 'D2') # run code and save to a file
D3 = calc_residual(D, 29.7987, 3) # data, beta value, order 3
save_result(D3, 'D3')

#Label name: S2 = Shiller order 2 
S2 = calc_residual(S, 17.3478, 2) # data, beta value, order 2             
save_result(S2, 'S2') # run code and save to a file
S3 = calc_residual(S, 17.4163, 3) # data, beta value, order 3
save_result(S3, 'S3')

#Label name: I2 = US Inflation order 2 
I2 = calc_residual(I, 0.0224, 2) # data, beta value, order 2             
save_result(I2, 'I2') # run code and save to a file
I3 = calc_residual(I, 0.0224, 3) # data, beta value, order 3
save_result(I3, 'I3')
 
def plot(label):
    QQdata = pd.read_csv(f"~/Dropbox/Fundamental Market Research/QQPlots/{label}.csv")
    numpyQQdata = QQdata.values
    newdata = numpy.array([numpyQQdata[k, 0] for k in range(numpy.size(numpyQQdata))])
    qqplot(newdata, line='s')
    pyplot.show()
    #pylab.title(f"Q-Q plot for {label}")
    #pylab.show()

plot("P2result")
plot("P3result")

plot("D2result")
plot("D3result")

plot("S2result")
plot("S3result")

plot("I2result")
plot("I3result")

#pylab.savefig('~/Downloads/P2result.png')

