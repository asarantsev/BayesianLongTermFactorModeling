#Bayesian Modeling, nominal S&P 500, using 1EY 

#Importing libraries
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import math
import matplotlib
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from numpy import linalg
from numpy import random
import statistics
from statsmodels import api

#Code for computing linear regressions.
#This code is adapted from https://github.com/asarantsev/BayesianLongTermFactorModeling
#which was written by the author "UNR Math Stat"
def correctLin(x, y):
    n = np.size(x)
    r = stats.linregress(x, y)
    s = r.slope
    i = r.intercept
    residuals = np.array([y[k] - x[k]*s - i for k in range(n)])
    stderr = math.sqrt((1/(n-2))*np.dot(residuals, residuals))
    print('normality', stats.shapiro(residuals)[1])
    print(r)
    return (s, i, stderr)

#Inverse chi squared
def ichi2(degFreedom, scale):
    shape = degFreedom/2
    return ((shape*scale)/random.gamma(shape))

#Bayesian simple linear regression
#y_i = slope * x_i + intercept + error_i
def BayesianRegression(x, y):
    n = len(x)
    M = [[np.dot(x, x), sum(x)], [sum(x), len(x)]]
    invM = linalg.inv(M)
    coeff = invM.dot([np.dot(x, y), sum(y)])
    slope = coeff[0]
    intercept = coeff[1]
    residuals = [y[k] - slope * x[k] - intercept for k in range(n)]
    var = np.dot(residuals, residuals)/(n-2)
    simVar = ichi2(n-2, var)
    simCoeff = random.multivariate_normal(coeff, invM*simVar)
    simSlope = simCoeff[0]
    simIntercept = simCoeff[1]
    return (simSlope, simIntercept, simVar)
#End adapted code

T = 1764

#Helper functions

#Returns two lists where both lists are actually defined
def intersect(list1, list2):
    listA = []
    listB = []
    time = min(len(list1), len(list2))
    for t in range(time):
        if (not math.isnan(list1[t])) and (not math.isnan(list2[t])):
            listA.append(list1[t])
            listB.append(list2[t])
    return [listA, listB]

def RNext(E_curr):
    epsilon = np.random.normal(0, sigma_E)
    E_next = a_E * E_curr + b_E + epsilon
    return E_next

def DNext(D_curr):
    epsilon = np.random.normal(0, sigma_D)
    D_next = a_D * D_curr + b_D + epsilon
    return D_next

#Equation (7)
#This function returns the ratio S(t+1)/S(t)
#based upon the current values P(t), D(t), R(t), and I(t) (passed as arguments)
#and r, c_P, c_D, c_R, c_I, sigma_S
def S_Ratio(E_curr):
    epsilon = np.random.normal(0, sigma)
    return math.exp(b +  a * E_curr + epsilon)




#Section 1: Data

#Getting data set up from files
#Data from 'Multpl.xlsx'
#Load file and set up data frames
#Note one will need to adjust the file path for their own computer.
file = pd.ExcelFile('Multpl.xlsx')
df = file.parse('Monthly')
#Create 2D list of data frame
multplData = df.values

#1D lists that will be better suited for our analyis
#E.g. DIVs[t] is equal to D(t)
CPIs = []
NOMRs = []
Rs = []
DIVs = []
SPs = []
RSPs = []
Ratios = []
EYs = []

#Parsing data from 2D list into better suited 1D lists
#Let us acquire real S&P returns given by
#Real S&P(t) = Nominal S&P(t) / CPI(t)
#We also get an array containing ln(S(t+1)/S(t))
for t in range(T+13):
    CPIs.append(multplData[T+12-t][1])
    SPs.append(multplData[T+12-t][5])
    RSPs.append(SPs[t] / CPIs[t])
    
for t in range(T+1):
    NOMRs.append(multplData[T-t][2])
    DIVs.append(multplData[T-t][4])
    EYs.append(multplData[T-t][3])
    appendee = NOMRs[t] - (CPIs[t] / CPIs[t-12] - 1)
    Rs.append(appendee)    
    
for t in range(T):
    Ratios.append(np.log(SPs[t+13]/SPs[t+12]))
    
    

    

    






#Section 5: Wealth Process

#At this point it is assumed that
# a_D, b_D, sigma_D,
# a_R, b_R, sigma_R,
# a, b, sigma
#are all declared and initialized with these exact names
#for the following functions.
def V(t, E_init, D_init):
    #Initializing variables
    V = 1
    E = E_init
    D = D_init

    for time in range(t):
        #Move V, R, and D forward according to models
        V *= S_Ratio(E)
        E = RNext(E)
        D = DNext(D)

        #If it's time to pay dividents, then pay dividents
        if (time + 1) % 12 == 0 and time + 1 >= 12:
            V *= (1 + D)

    return V



#Section 6: Simulation
NSIMS = 10000
Vs = []
A_E = []
B_E = []
S_E = []
A_D = []
B_D = []
S_D = []
A_S = []
B_S = []
S_S = []

print('AR(1) for EY')
print('')
print(correctLin(EYs[:T], [EYs[k+1]-EYs[k] for k in range(T)]))
print('')
print('AR(1) for EY')
print('')
print(correctLin(EYs[:T], EYs[1:]))
print('')
print('AR(1) for D')
print('')
print(correctLin(DIVs[:T], [DIVs[k+1] - DIVs[k] for k in range(T)]))
print('')
print('AR(1) for D')
print('')
print(correctLin(DIVs[:T], DIVs[1:]))
print('')
print('regression for price returns vs EY')
print('')
print(correctLin(EYs[:T], Ratios))
print('')
print('regression for price returns vs D')
print('')
print(correctLin(DIVs[:T], Ratios))
print('')

X = pd.DataFrame({'R': DIVs[:T]})
X = api.add_constant(X)
Reg = api.OLS(Ratios, X).fit()
print(Reg.rsquared)

for i in range(NSIMS):
    #Getting lists where all elements are defined.
    #Simulating regression parameters using BayesianRegression function
    a_E, b_E, sigma_E = BayesianRegression(EYs[:T], EYs[1:])
    a_D, b_D, sigma_D = BayesianRegression(DIVs[:T], DIVs[1:])
    a, b, sigma = BayesianRegression(EYs[:T], Ratios)
    sigma_E **= (1/2)
    sigma_D **= (1/2)
    sigma **= (1/2)
    
    A_E.append(a_E)
    B_E.append(b_E)
    S_E.append(sigma_E)
    A_D.append(a_D)
    B_D.append(b_D)
    S_D.append(sigma_D)
    A_S.append(a)
    B_S.append(b)
    S_S.append(sigma)

    
    
def simul(NMONTHS, initEY, initDIV):
    ret = []
    for sim in range(NSIMS):
        a_E = A_E[sim]
        b_E = B_E[sim]
        sigma_E = S_E[sim]
        a_D = A_D[sim]
        b_D = B_D[sim]
        sigma_D = S_D[sim]
        a = A_S[sim]
        b = B_S[sim]
        sigma = S_S[sim]
        wealth = V(NMONTHS, initEY, initDIV)
        ret.append(math.log(wealth)*(12/NMONTHS))
    return ret   

print('intercept E')
pyplot.hist(A_E, bins = 100)
pyplot.show()
print('slope E')
pyplot.hist(B_E, bins = 100)
pyplot.show()
print('variance E')
pyplot.hist(S_E, bins = 100)
pyplot.show()
print('intercept D')
pyplot.hist(A_D, bins = 100)
pyplot.show()
print('slope D')
pyplot.hist(B_D, bins = 100)
pyplot.show()
print('variance D')
pyplot.hist(S_D, bins = 100)
pyplot.show()
print('intercept S')
pyplot.hist(A_S, bins = 100)
pyplot.show()
print('slope S')
pyplot.hist(B_S, bins = 100)
pyplot.show()
print('variance S')
pyplot.hist(S_S, bins = 100)
pyplot.show()

 
    
ret = simul(120, EYs[T], DIVs[T])
print('')
print("Current Data as Initial Data")
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Average:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

meanEYs = np.mean(EYs)
meanDIVs = np.mean(DIVs)

ret = simul(120, meanEYs, meanDIVs)
print('')
print("Long-Term Averages as Initial Data")
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Average:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(60, EYs[T], DIVs[T])
print('')
print("Current Data as Initial Data")
print("With time of", 60, " months and ", NSIMS, " simulations")
print("")
print("Average:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(60, meanEYs, meanDIVs)
print('')
print("Long-Term Averages as Initial Data")
print("With time of", 60, " months and ", NSIMS, " simulations")
print("")
print("Average:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, 0.6*meanEYs, meanDIVs)
print('')
print("Initial Data: EY and DIV", 0.6, 1.0)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, 0.8*meanEYs, meanDIVs)
print('')
print("Initial Data: EY and DIV", 0.8, 1.0)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, 1.2*meanEYs, meanDIVs)
print('')
print("Initial Data: EY and DIV", 1.2, 1.0)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, 1.4*meanEYs, meanDIVs)
print('')
print("Initial Data: EY and DIV", 1.4, 1.0)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, 1.6*meanEYs, meanDIVs)
print('')
print("Initial Data: EY and DIV", 1.6, 1.0)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, meanEYs, 0.6*meanDIVs)
print('')
print("Initial Data: EY and DIV", 1.0, 0.6)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, meanEYs, 0.8*meanDIVs)
print('')
print("Initial Data: EY and DIV", 1.0, 0.8)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, meanEYs, 1.2*meanDIVs)
print('')
print("Initial Data: EY and DIV", 1.0, 1.2)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, meanEYs, 1.4*meanDIVs)
print('')
print("Initial Data: EY and DIV", 1.0, 1.4)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))

ret = simul(120, meanEYs, 1.6*meanDIVs)
print('')
print("Initial Data: EY and DIV", 1.0, 1.6)
print("With time of", 120, " months and ", NSIMS, " simulations")
print("")
print("Averages:", statistics.mean(ret))
print("")
print("Stdev:", statistics.stdev(ret))
print("")
print("95% value at risk:", np.percentile(ret, 5))