# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy
import pandas
import scipy
from scipy import stats
from pandas import DataFrame
#import linmodelclass
#from linmodelclass import Linmodel
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt

#Comment by Taran:
#At this point it is assumed that
#a_P, b_P, sigma_P,
# a_D, b_D, sigma_D,
# a_R, b_R, sigma_R,
# a_I, b_I, sigma_I,
# r, c_P, c_D, c_R, c_I, sigma_S
#Ir, Ic_P, Ic_D, Ic_R, Ic_i, Isigma_S (Inflation Version)
#are all declared and initialized with these exact names
#for the following functions.

#Equations (2) and (3)
#This function returns the value of P, D, R, or I at a given time, t,
#based upon the current value P(0), D(0), R(0), or I(0),
#the corresponding coefficients a, b, and sigma.
def PDRI(t, PDRI_curr, a_PDRI, b_PDRI, sigma_PDRI):
    #Initializing some variables
    PDRI_next = PDRI_curr

    #Use eq. (2) and (3) t times to determine the current value at time, t,
    for i in range (t):
        #Generate residual epsilon according to given sigma
        epsilon = numpy.random.normal(0, sigma_PDRI)

        #Compute next and store it as current
        PDRI_next = a_PDRI * PDRI_next + b_PDRI + epsilon

    return PDRI_next

#Equation (7)
#This function returns the ratio S(t+1)/S(t)
#based upon the current values P(t), D(t), R(t), and I(t) (passed as arguments)
#and r, c_P, c_D, c_R, c_I, sigma_S
def S_Ratio(P, D, R, I):

    #We create a residual from a normal distribution
    epsilon = numpy.random.normal(0, sigma_S)
    #print(epsilon)
    #We now retuirn S(t+1)/S(t)
    return math.exp(r + c_P * P + c_D * D + c_R * R + c_I * I + epsilon)

#Equation (8)
#This function returns the inflation adjusted
#ratio S*(t+1)/S*(t) based upon
#the current values P(t), D(t), R(t), and I(t) (passed as arguments)
#and Ir, Ic_P, Ic_D, Ic_R, Ic_i, Isigma_S
def IS_Ratio(P, D, R, I):

    #We create a residual from a normal distribution
    epsilon = numpy.random.normal(0, Isigma_S)
    #print(epsilon)
    #We now retuirn S(t+1)/S(t)
    return math.exp(Ir + Ic_P * P + Ic_D * D + Ic_R * R + Ic_I * I + epsilon)

#This function returns the terminal wealth V(t, w, l, k) at a given time, t.
#Here w, l, k are given to simulate the wealth process.
#We must also input P(0), D(0), R(0), I(0).
#Again, the other variables should be globally declared.
def V(t, w, l, k,
      P_curr, D_curr, R_curr, I_curr):

    #Initialize variables
    V = 1
    P = P_curr
    D = D_curr
    R = R_curr
    I = I_curr
    nom_div = D_curr
    #List of inflations every 12 months
    I12s = []

    #This loop ultimately gives S_ratio = S(t) / S(0)
    for time in range(t):

        #Multiply V by the current ratio
        V *= S_Ratio(P, D, R, I)

        #Get the next P(t), D(t), R(t), I(t)
        P = PDRI(1, P, a_P, b_P, sigma_P)
        D = PDRI(1, D, a_D, b_D, sigma_D)
        R = PDRI(1, R, a_R, b_R, sigma_R)
        I = PDRI(1, I, a_I, b_I, sigma_I)

        #If it has been a year, adjust then pay dividents
        #Note we use 11 since time is of the form, 0,1,2,...,t-1
        #Thus if it is the tweflth time step then time is 11.
        if (time + 1) % 12 == 0 and time + 1 >= 12:
            I12s.append(I)
            nom_div = 1 / D

            for I in I12s:
                nom_div *= I

            V *= (1 + nom_div)

        #Only if we are retired, and it is the kth month should we withdraw w
        #We use time + 1 since time = 0,1,2,...,t-1
        if time + 1 >= l and (time + 1 - l) % k == 0 and time + 1 > 0:
            V -= w

        #Check for bankruptcy at every time step
        if V <= 0:
            return 0

    return V

#This function returns the terminal wealth V*(t, w, l, k)
#as the function above does but instead adjusts for inflation.
#Recall we must use the correct coefficients
#r*, c*_P, c*_D, c*_R, c*_I, and sigma_S*
def IV(t, w, l, k,
      P_curr, D_curr, R_curr, I_curr):

    #Initialize variables
    IV = 1
    P = P_curr
    D = D_curr
    R = R_curr
    I = I_curr

    #This loop ultimately gives S_ratio = S(t) / S(0)
    for time in range(t):

        #Multiply V by the current ratio
        IV *=  IS_Ratio(P, D, R, I)

        #Get the next P(t), D(t), R(t), I(t)
        P = PDRI(1, P, a_P, b_P, sigma_P)
        D = PDRI(1, D, a_D, b_D, sigma_D)
        R = PDRI(1, R, a_R, b_R, sigma_R)
        I = PDRI(1, I, a_I, b_I, sigma_I)

        #If it has been a year, then adjust w for inflation and pay dividents
        if (time + 1) % 12 == 0 and t + 1 >= 12:
            #This is just multiplying w by 1 + I(t) * w
            w *= (1 + I)
            IV *= 1 + ( 1 / D)

        #Only if we are retired, and it is the kth month should we withdraw w
        #We use time + 1 since time = 0,1,2,...,t-1
        if time + 1 >= l and (time + 1 - l) % k == 0 and time + 1 > 0:
            IV = IV - w

        #Check for bankruptcy at every time step
        if IV <= 0:
            return 0

    return IV

#This function returns a list of non-inflation adjusted terminals wealths
#at time, t.
#It also computes the probability of bankruptcy, value at risk,
#and creates a histogram of the terminal wealths.
def step6(t, w, l, k, alpha,
          P_curr, D_curr, R_curr, I_curr):

    #Initialize variables
    num_Bankrupt = 0
    final_Wealths = []

    #Create a list of terminal wealths
    for i in range(0, 10000):
        final_Wealths.append(V(t, w, l, k,
                               P_curr, D_curr, R_curr, I_curr))

    #Count how many went bankrupt
    for wealth in final_Wealths:
        if wealth == 0:
            num_Bankrupt += 1

    #Only if none went bankrupt
    if num_Bankrupt == 0:
        #Create and output histogram
        plt.hist(final_Wealths)
        plt.show()

    #Compute and output the probability of bankruptcy
    prob_Bankrupt = num_Bankrupt / len(final_Wealths)
    print("The probability of bankruptcy is", prob_Bankrupt)
    print()

    #Sort final wealths, compute the index, and output the value at risk
    final_Wealths.sort()
    index = int( (1 - alpha) * len(final_Wealths) )
    print("The value at risk is", final_Wealths[index],
    "for a confidence level of", alpha)

    return final_Wealths

#This function returns a list of inflation adjusted terminals wealths
#at time, t.
#It also computes the probability of bankruptcy, value at risk,
#and creates a histogram of the terminal wealths.
#Simply use inflation coefficients
#r*, c*_P, c*_D, c*_R, c*_I, and sigma_S*
def step7(t, w, l, k, alpha,
          P_curr, D_curr, R_curr, I_curr):

    #Initialize variables
    num_Bankrupt = 0
    final_Wealths = []

    #Create a list of terminal wealths
    for i in range(0, 10000):
        final_Wealths.append(IV(t, w, l, k,
                               P_curr, D_curr, R_curr, I_curr))

    #Count how many went bankrupt
    for wealth in final_Wealths:
        if wealth == 0:
            num_Bankrupt += 1

    #Only output if none went bankrupt
    if num_Bankrupt == 0:
        #Create and output histogram
        plt.hist(final_Wealths, bins = 100)
        plt.show()

    #Compute and output the probability of bankruptcy
    prob_Bankrupt = num_Bankrupt / len(final_Wealths)
    print("The probability of bankruptcy is", prob_Bankrupt)
    print()

    #Sort final wealths, compute the index, and output the value at risk
    final_Wealths.sort()
    index = int( (1 - alpha) * len(final_Wealths) )
    print("The value at risk is", final_Wealths[index],
    "for a confidence level of", alpha)

    return final_Wealths

#This functions finds, prints, and returns the probability
#that the S&P 500 outperforms the 10 year T-Note.
#This function works however, it takes a very long time
#to do that many computations if time = 120.
def step8(P_curr, D_curr, R_curr, I_curr):

    #The number of terminal wealths that outperform 10-year T-note.
    num_Better = 0
    final_Wealths = []

    #Create a list of terminal wealths
    for i in range(0, 10000):
        final_Wealths.append(V(120, R_curr / 2, 0, 6,
                               P_curr, D_curr, R_curr, I_curr))

    #Count how many terminals wealths outperformed the 10-year T-note.
    for wealth in final_Wealths:
        if wealth > 1:
            num_Better += 1

    #Compute and output the probability
    prob_Better = num_Better / len(final_Wealths)
    print("The probability of outperformance is", prob_Better)

    return prob_Better

#The code above was written by Taran Grove between February 25 - March 5
#and slightly edited by Andrey Sarantsev on March 8

#TESTING
#Variables that must be declared to use program
r = 0
c_P = 0
c_D = 0
c_R = 0
c_I = 0
sigma_S = 0.001

Ir = 0
Ic_P = 1
Ic_D = 1
Ic_R = 1
Ic_I = 1
Isigma_S = 0.001

a_P = 1
b_P = 0
sigma_P = 0.001

a_D = 1
b_D = 0
sigma_D = 0.001

a_R = 1
b_R = 0
sigma_R = 0.001

a_I = 1
b_I = 0
sigma_I = 0.001

print("V Test")
print(V(15, .1, 7, 3, 1, 1, 1, 1))
print()
print("Step 6 Test")
step6(12, 0.2, 7, 3, 0.99, 1, 1, 1, 1)
print()
print("Step 7 Test")
step7(12, 0.2, 7, 3, 0.99, 1, 1, 1, 1)
print()
print("Step 8 Test")
step8(1, 1, 1, 1)
