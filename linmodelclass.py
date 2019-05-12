#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import stats

from statsmodels.graphics.gofplots import qqplot 

import numpy

import matplotlib.pyplot as plt
#Akram: This is a class that will hold all our data so we can do the analysis from here.
class Linmodel(): 
    def __init__ (self, data, timestep):
        self.data = data
        self.timestep = timestep
        self.autoregress1()
        self.average_of_data()
        self.residoo()
        
    def autoregress1(self): #autoregression of order one, x is lapsed by one, y is the whole data
        self.slope, self.intercept, _, _, _ = stats.linregress(self.data[1:], self.data[:self.timestep-1])
        
    def average_of_data(self): #Finds mean
        i = 0
        self.sum_of_data = 0
        while i < self.timestep:
            self.sum_of_data = self.sum_of_data + self.data[i]
            i = i+1
        self.mean_of_data = self.sum_of_data / self.timestep
    
    def residoo(self):#returns an array containing the residues of our data
        self.residual_of_data = numpy.empty_like(self.data) #using this data type to stay consistent with rest of our data types
        self.average_of_data()
        i = 1
        while i < self.timestep:
            self.observed = self.data[i]
            self.predicted = (-1) * (1 - self.slope) * self.data[i-1] + (self.slope * self.mean_of_data) # -(feedback)*(previous value) + (the slope * our average)
            self.residual_of_data[i] = self.observed - self.predicted
#            print(self.observed - self.predicted)
            i = i+1
    
    def Q_Q_plot(self):
        qqplot(self.residual_of_data, line = 's') 
        plt.show()