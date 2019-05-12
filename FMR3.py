# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:01:07 2019

@author: UNR Math Stat
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:49:13 2019

@author: UNR Math Stat
"""

import numpy
import matplotlib
import pandas
import math
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
import statsmodels
from statsmodels import api
import scipy
from scipy import stats
from decimal import Decimal


#read the data
dataframe = pandas.read_excel('newfindata.xlsx', sheet_name = 'Updated')
data = dataframe.values


cpi = data[:, 0]#current consumer price index
T = numpy.size(cpi)#number of time points in raw data
N = T - 120#number of time points in prepared data
rate = data[:, 1]#10-year nominal Treasury interest rate
shiller = data[:, 2]#10-year trailing price-ro-earnings Shiller ratio
dy1 = data[:T-1, 3]#1-year trailing dividend yield
sp500 = data[:, 4]# S&P 500 nominal prices

#This is inflation rate averaged over the last 10 years
ir = [0.1*(cpi[t]/cpi[t+120] - 1) for t in range(N)]
#This is real Trasury interest rate, 10-year
tr = [(rate[k] + 1)/(ir[k] + 1) - 1 for k in range(N)]
#This is 10-year trailing earnings yield
ey = 1/shiller[:N]
#This is real (inflation-adjuste) S&P 500 prices
sp = [sp500[k]/cpi[k] for k in range(T)]
#This is trailing 10-year dividend yield
dy = numpy.array([numpy.mean([dy1[t+12*k]*sp[t+12*k] for k in range(10)])/sp[t] for t in range(N)])
#Monthly real S&P 500 returns
ret = [math.log(sp[k]/sp[k+1]) for k in range(N-1)]

ret12 = [math.log(sp[k]/sp[k+120])/10 for k in range(N-1)]

#This is the correct function for simple linear regression.
#Built-in function for standard error returned standard error for 
#the slope, not for the sigma in the error term.
#So we have to manually write it.
#Commented out are QQ plots, used for normality testing of residuals.
def correctLin(x, y):
    n = numpy.size(x)
    r = stats.linregress(x, y)
    s = r.slope
    i = r.intercept
    print(r)
    residuals = numpy.array([y[k] - x[k]*s - i for k in range(n)])
    stderr = math.sqrt((1/(n-2))*numpy.dot(residuals, residuals))
    qqplot(residuals, line = 'r')
    pyplot.show()
    pyplot.plot(residuals)
    pyplot.show()
    print('normality', stats.shapiro(residuals))
    return (s, i, stderr)

def simpleLin(x, y):
    n = numpy.size(x)
    x = numpy.array(x)
    y = numpy.array(y)
    k = numpy.dot(x, y)/numpy.dot(x, x)
    residuals = y - k*x
    stderr = numpy.std(residuals)
    qqplot(residuals, line = 'r')
    pyplot.show()
    pyplot.plot(residuals)
    pyplot.show()
    print('normality', stats.shapiro(residuals))
    return (k, stderr)

def ARSV(x, k, a, p):
    n = numpy.size(x)
    mx = [item - numpy.mean(x) for item in x]
    vol = [(mx[k] - mx[k+1])**2 for k in range(n-1)]
    mavol = [math.sqrt(a*numpy.mean(vol) + numpy.mean(vol[i:i+k-1])) 
        for i in range(n-k)]
    pyplot.plot(mavol)
    return simpleLin([mx[i]/(mavol[i+1]**p) for i in range(n-k-1)], 
                  [mx[i+1]/(mavol[i+1]**p) for i in range(n-k-1)])
    
def GARSV(x, k, a, p):
    n = numpy.size(x)
    mx = [item - numpy.mean(x) for item in x]
    vol = [(mx[k] - mx[k+1])**2 for k in range(n-1)]
    mavol = [math.sqrt(a*numpy.mean(vol) + numpy.mean(vol[i:i+k-1])) 
        for i in range(n-k)]
    pyplot.plot(mavol)
    pyplot.show()
    return correctLin([mx[i]/(mavol[i+1]**p) for i in range(n-k-1)], 
                  [mx[i+1]/(mavol[i+1]**p) for i in range(n-k-1)])
    
def NARSV(x, k, a, p, q):
    n = numpy.size(x)
    mx = [item - numpy.mean(x) for item in x]
    vol = [(mx[k] - mx[k+1])**2 for k in range(n-1)]
    mavol = [math.sqrt(a*numpy.mean(vol) + numpy.mean(vol[i:i+k-1])) 
        for i in range(n-k)]
    pyplot.plot(mavol)
    pyplot.show()
    X1 = [mx[i+1]/(mavol[i+1]**p) for i in range(n-k-1)]
    X2 = [item**(q-p) for item in mavol[1:n-k]]
    #X3 = [item**r for item in mavol[1:n-k]]
    Y = [mx[i]/(mavol[i+1]**p) for i in range(n-k-1)]
    return Regression2(X1, X2, Y)
                  
    


#This function performs AR(1) on the array x, assuming x[0] is the 
#latest data point; that is, values are in backward order,
#as in our financial data
def qqres(x):
    n = numpy.size(x)
    return correctLin(x[1:], x[:n-1])

def Regression2(X1, X2, Y):
    n = numpy.size(Y)
    X = pandas.DataFrame({'1': X1, '2': X2})
    X = api.add_constant(X)
    Reg = api.OLS(Y, X).fit()
    Y_Predictions = Reg.predict(X)
    print(Reg.summary())
    #print([Decimal(Reg.params[k]) for k in range(3)])
    residuals = Y - Y_Predictions
    stderr = math.sqrt(1/(n-3)*numpy.dot(residuals, residuals))
    print('normality', stats.shapiro(residuals))
    qqplot(residuals, line = 's')
    pyplot.show()
    return (stderr)
     
#This is multiple lineear regression of Y over X1, X2, X3.
#Prints summary, returns standard error, and prints coefficients
#with many decimals. We also commented out the QQ plot.
def Regression3(X1, X2, X3, Y):
    n = numpy.size(Y)
    X = pandas.DataFrame({'1': X1, '2': X2, '3' : X3})
    X = api.add_constant(X)
    Reg = api.OLS(Y, X).fit()
    Y_Predictions = Reg.predict(X)
    print(Reg.summary())
    #print([Decimal(Reg.params[k]) for k in range(4)])
    residuals = Y - Y_Predictions
    stderr = math.sqrt(1/(n-4)*numpy.dot(residuals, residuals))
    print('normality', stats.shapiro(residuals))
    pyplot.plot(residuals)
    pyplot.show()
    qqplot(residuals, line = 's')
    pyplot.show()
    return (stderr)

#This is the covariance function for random samples a and b
def cov(a, b):
    n = numpy.size(a)
    ma = numpy.mean(a)
    mb = numpy.mean(b)
    c = 1/(n-1)*sum([(a[k] - ma)*(b[k] - mb) for k in range(n)])
    return (c)

#This is 3D autoregression AR(1), which I tried to use for joint
#modeling of 10-year earnings yield, 10-year divided yield,
#and 10-year real interest rate. 
#This is not important for now, but we might use it later.
def ar3(X1, X2, X3):
    n = numpy.size(X1)
    X = pandas.DataFrame({'1': X1[1:], '2': X2[1:], '3': X3[1:]})
    X = api.add_constant(X)
    Reg1 = api.OLS(X1[:n-1], X).fit()
    print(Reg1.summary())
    res1 = X1[:n-1] - Reg1.predict(X)
    Reg2 = api.OLS(X2[:n-1], X).fit()
    print(Reg2.summary())
    res2 = X2[:n-1] - Reg2.predict(X)
    Reg3 = api.OLS(X3[:n-1], X).fit()
    print(Reg3.summary())
    res3 = X3[:n-1] - Reg3.predict(X)
    covmatrix = [[cov(res1, res1), cov(res1, res2), cov(res1, res3)], 
                  [cov(res2, res1), cov(res2, res2), cov(res2, res3)],
                  [cov(res3, res1), cov(res3, res2), cov(res3, res3)]]
    #print([Decimal(Reg1.params[k]) for k in range(4)])
    #print([Decimal(Reg2.params[k]) for k in range(4)])
    #print([Decimal(Reg3.params[k]) for k in range(4)])
    return covmatrix
    



    

