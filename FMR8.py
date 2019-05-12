# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:00:19 2019

@author: UNR Math Stat
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:58:04 2019

@author: UNR Math Stat
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:49:40 2019

@author: UNR Math Stat
"""


# Author: Taran Grove and Andrey Sarantsev
# Start Date: 4/23/2019
# Purpose: This code implements the dynamic factor model 
# described in Section 4.3 in the article 
# LONG-TERM FACTOR INVESTING WITH BAYESIAN STATISTICS 
# written by Akram Reshad, Taran Grove, Andrey Sarantsev.

# We simulate S&P 500 wealth process with dividends paid annually,
# with withdrawals equal to 10-year real interest rate,
# divided by 2, semiannually, for 10 years.  
# This enables us to compare S&P 500 with 10-year inflation-adjusted T-bond.
# We find the probability of S&P 500 outperforming the bond 

# Before this, we fit the model: regression of monthly returns and 
# next year's dividend yield on the current 10-year dividend yield,
# current 10-year earnings yield, and 10-year real interest rate,
# estimated using the last 10 year inflation rate. 
# The three factors are modeled using AR(1) with diagonal covariance matrix.

# We use Bayesian statistics to estimate coefficients of regression,
# starting from non-informative prior. This helps us account for uncertainty.


import numpy
from numpy import random
from numpy import linalg
import math
import pandas
import scipy
from scipy import stats
import matplotlib
from matplotlib import pyplot
import statsmodels
from statsmodels.graphics.gofplots import qqplot
from statsmodels import api

# reading the data
dataframe = pandas.read_excel('newfindata.xlsx', sheet_name = 'Updated')
data = dataframe.values

# splitting raw data into columns
cpi = data[:, 0] # current consumer price index
T = numpy.size(cpi) # number of time points in raw data
N = T - 120 # number of time points in prepared data
rate = data[:, 1] # 10-year nominal Treasury interest rate
shiller = data[:, 2] # 10-year trailing price-ro-earnings Shiller ratio
dy1 = data[:T-1, 3] # 1-year trailing dividend yield
sp500 = data[:, 4] # S&P 500 nominal prices

# Creation of data for analysis
# This is inflation rate averaged over the last 10 years
ir = [0.1*(cpi[t]/cpi[t+120] - 1) for t in range(N)]
# This is real Treasury interest rate, 10-year
tr = [(rate[k] + 1)/(ir[k] + 1) - 1 for k in range(N)]
# This is 10-year trailing earnings yield
ey = 1/shiller[:N]
# This is real (inflation-adjusted) S&P 500 prices
sp = [sp500[k]/cpi[k] for k in range(T)]
# This is trailing 10-year dividend yield
dy = numpy.array([numpy.mean([dy1[t+12*k]*sp[t+12*k] for k in range(10)])/sp[t] for t in range(N)])
# Monthly real S&P 500 returns
ret = [math.log(sp[k]/sp[k+1]) for k in range(N-1)]

# Implementation of 3-dimensional autoregression of order 1 to simulate DY, EY, TR.
# This functions returns F(t), based on global variables for Equation (2)
# in 'parameters' and a given F(t-1) vector.
def F(F_prev, parameters):
    # Compute epsilon vector based on covariance matrix
    a = parameters[0]
    b = parameters[1]
    E = parameters[2]
    epsilon = random.multivariate_normal((0,0,0), E)
    
    # Compute F(t) according to Equation (2)
    F_next = a.dot(F_prev) + b + epsilon

    return F_next

# Implementation of autoregression to simulate to simulate next year's DY.
# This function returns Delta(t+12), based on global variables 
# in 'parameters' and a given D(t).
def Delta(D_curr, E_curr, R_curr, parameters):
    # Compute little delta based on the variance of little delta.
    DeltaCoeff = parameters[3]
    std_Delta = parameters[4]
    delt = random.normal(0, std_Delta)
    
    # Compute Delta(t + 12) according to Equation (3)
    Delta = DeltaCoeff[0] + DeltaCoeff[1] * D_curr + DeltaCoeff[2] * E_curr + DeltaCoeff[3] * R_curr + delt

    return Delta

# Equation (4)
# Implementation of multiple regression to simulate S&P 500 monthly returns.
# This functions returns S(t+1)/S(t), based on global variables 
# and given D(t), E(t), and R(t)
def S_Ratio(D_curr, E_curr, R_curr, parameters):
    # Compute epsilon based on variance of S
    SPCoeff = parameters[5]
    std_S = parameters[6]
    epsilon = random.normal(0, std_S)
    r = SPCoeff[0]
    alpha_D = SPCoeff[1]
    alpha_E = SPCoeff[2]
    alpha_R = SPCoeff[3]
    # Compute S(t+1)/S(t) according to Equation (4)
    ratio = math.exp(r + alpha_D * D_curr + alpha_E * E_curr
                     + alpha_R * R_curr + epsilon)

    return ratio


# Implementation of the wealth process.
# This function returns V(t), based on a given time t,
# and global variables for Equations (2), (3), and (4) respectively.
# Withdrawals are semiannual, every 6 months
# We start with wealth 1 and automatically reinvest dividends,
# computed using forward dividend yield 
def V(t, D_curr, E_curr, R_curr, withdrawal, parameters):

    # Initializing some variables
    V = 1
    Evol = numpy.array([1])
    F_vect = numpy.array([D_curr, E_curr, R_curr])
    #This list will store Delta(12), Delta(24), Delta(36), ...
    Delta_12s = []

    for time in range(t):
        #Multipy by the current ratio
        V *= S_Ratio(D_curr, E_curr, R_curr, parameters)

        #Store Delta(12), Delta(24), Delta(36), ...
        if time % 12 == 0:
            Delta_12s.append(Delta(D_curr, E_curr, R_curr, parameters))

        #If it has been a year, then pay dividends
        #Note we use time + 1 since time is of the form, 0,1,2,...,t-1
        if (time + 1) % 12 == 0:
            index = ((time + 1) / 12) - 1
            index = int(index)
            V *= (1 + Delta_12s[index])
            
        if (time + 1) % 6 == 0:
            V = V - withdrawal

        #Advance F vector
        F_vect = F(F_vect, parameters)
        Evol = numpy.append(Evol, V)
    #pyplot.plot(Evol)
    return V

# Simulation of the inverse chi-squared distribution
# Which serves as posterior distribution for variance 
def ichi2(degFreedom, scale):
    shape = degFreedom/2
    return ((shape*scale)/random.gamma(shape))

ic = numpy.ones(N) # Vector of units
Factors = numpy.column_stack([ic, dy, ey, tr]) # Factor matrix
Past = Factors[1:][:] # Past observations
TPast = numpy.transpose(Past) 
C = numpy.matmul(TPast, Past)
I = numpy.linalg.inv(C)
M = numpy.matmul(I, TPast)
dF = N - 5 # Degrees of freedom: N-1 = number of time points, 4 = parameters

# Regression given target variable
def regress(target):
    coeff = M.dot(target) #regression coefficients
    residuals = target - Past.dot(coeff) 
    sigma2 = numpy.dot(residuals, residuals) / dF # estimate of variance
    return (coeff, sigma2)

# Bayesian Regression
def BayesSim(target):
    coeff, sigma2 = regress(target) 
    simVar = ichi2(dF, sigma2) # Simulated variance
    simCoeff = random.multivariate_normal(coeff, simVar*I) # Sim coefficient
    return (simCoeff, simVar)

# Point estimates a, b, E 
a = numpy.array([
        [.9875, 0.005433, 0.005145],
        [-0.000083, .9948, 0.003490],
        [0.001586, -0.008593, .9919],
        ]) 

b = (10 ** (-5)) * numpy.array(
    [4.088, 29.92, 62.44]
    )
E = (10 ** (-6)) * numpy.array([
    [2.227**2, 0, 0],
    [0, 3.605**2, 0],
    [0, 0, 2.22**2]
    ])

# Long-term averages of factors from AR(3)
means = linalg.inv(numpy.identity(3) - a).dot(b)

# Point Estimates for Regression of forward 1-year diviend yield
D_Delta = .9246
E_Delta = -0.1035
R_Delta = 0.0101
i_Delta = 0.0112
std_Delta = 0.007418

# Point Estimates for Regression of 1-month future S&P 500 returns
r = 0.0032
alpha_D = 0.0215
alpha_E = -0.0169
alpha_R = -0.0803
std_S = 0.04132


NSIMS = 1000 # Number of simulations
dyMean = numpy.mean(dy) # long-term average of DY; close to means[0]
eyMean = numpy.mean(ey) # Long-term average pf EY; close to means[1]
trMean = numpy.mean(tr) # Long-term average of TR; close to means[2]

# In these arrays we write our simulated regression parameters
# sampled from posterior distribution
EYSimVar = numpy.array([]) # Variance for regression for EY
DYSimVar = numpy.array([]) # Variance for regression for DY
TRSimVar = numpy.array([]) # Variance for regression for TR
SP500SimVar = numpy.array([]) # Variance for regression for S&P returns
fwdDYSimVar = numpy.array([]) # Variance for forward 1-year DY
EYSimCoeff = numpy.array([]) # Coefficients for regression for EY
DYSimCoeff = numpy.array([])
TRSimCoeff = numpy.array([])
SP500SimCoeff = numpy.array([])
fwdDYSimCoeff = numpy.array([])

# Here we simulate these regression coefficients
for sim in range(NSIMS):
    simCoeff, simVar = BayesSim(ey[:N-1])
    EYSimVar = numpy.append(EYSimVar, simVar)
    EYSimCoeff = numpy.append(EYSimCoeff, simCoeff)
    simCoeff, simVar = BayesSim(dy[:N-1])
    DYSimVar = numpy.append(DYSimVar, simVar)
    DYSimCoeff = numpy.append(DYSimCoeff, simCoeff)
    simCoeff, simVar = BayesSim(tr[:N-1])
    TRSimVar = numpy.append(TRSimVar, simVar)
    TRSimCoeff = numpy.append(TRSimCoeff, simCoeff)
    simCoeff, simVar = BayesSim(ret[:N-1])
    SP500SimVar = numpy.append(SP500SimVar, simVar)
    SP500SimCoeff = numpy.append(SP500SimCoeff, simCoeff)
    simCoeff, simVar = BayesSim(dy1[:N-1])
    fwdDYSimVar = numpy.append(fwdDYSimVar, simVar)
    fwdDYSimCoeff = numpy.append(fwdDYSimCoeff, simCoeff)

# List of possible initial values for DY, EY, TR
dyInitialList = numpy.array([])
eyInitialList = numpy.array([])
trInitialList = numpy.array([])

# Probability that S&P 500 wealth + withdrawals beats 10-year TIPS
def comparison(dy0, ey0, tr0):
    results = 0
    for sim in range(NSIMS):
        # We take simulated coefficients and write them into 'parameters'
        slopes = []
        slopes = numpy.append(slopes, DYSimCoeff[4*sim+1:4*sim+4])
        slopes = numpy.append(slopes, EYSimCoeff[4*sim+1:4*sim+4])
        slopes = numpy.append(slopes, TRSimCoeff[4*sim+1:4*sim+4])
        # a is the 3x3 matrix
        a = numpy.split(slopes, 3)
        a = numpy.array(a)
        # b is the intercept vector
        b = numpy.array([DYSimCoeff[4*sim], EYSimCoeff[4*sim], TRSimCoeff[4*sim]])
        # E is diagonal covariance matrix for AR(1) residuals in 3D
        E = numpy.diagflat([DYSimVar[sim], EYSimVar[sim], TRSimVar[sim]])
        DeltaCoeff = fwdDYSimCoeff[4*sim:4*sim+4]
        std_Delta = math.sqrt(fwdDYSimVar[sim])
        SPCoeff = SP500SimCoeff[4*sim:4*sim+4]
        std_S = math.sqrt(SP500SimVar[sim])
        parameters = [a, b, E, DeltaCoeff, std_Delta, SPCoeff, std_S]
        # wealth process 120 months later with given initial DY, EY, TR
        # semiannual withdrawals are equal to half of TR
        wealth = V(120, dy0, ey0, tr0, tr0/2, parameters)
        if (wealth > 1): 
            results = results + 1
    return (results / NSIMS)

print('mean values = ', comparison(dyMean, eyMean, trMean))
print('current values = ', comparison(dy[0], ey[0], tr[0]))

# Simulate NITEMS of initial possible DY, EY, TR
NITEMS = 100
dyValues = [k * dyMean for k in random.uniform(0.6, 1.8, NITEMS)]
eyValues = [k * eyMean for k in random.uniform(0.6, 1.8, NITEMS)]
trValues = [k * trMean for k in random.uniform(0.6, 1.8, NITEMS)]
Probs = [] # Here we shall record probabilities

for item in range(NITEMS):
    print('dy = ', dyValues[item])
    print('ey = ', eyValues[item])
    print('tr = ', trValues[item])
    currentProb = comparison(dyValues[item], eyValues[item], trValues[item])
    Probs.append(currentProb)
    print(currentProb)
    
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
    residuals = Y - Y_Predictions
    stderr = math.sqrt(1/(n-4)*numpy.dot(residuals, residuals))
    print('normality', stats.shapiro(residuals))
    pyplot.plot(residuals)
    pyplot.show()
    qqplot(residuals, line = 's')
    pyplot.show()
    return (stderr)

# run linear regression of probability of S&P 500 beating 10Y TIPS
# versus initial DY, EY, TR
Regression3(dyValues, eyValues, trValues, Probs)
