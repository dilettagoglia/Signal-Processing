# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:31:46 2020

@author: Diletta
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt 
from hmmlearn.hmm import GaussianHMM 
from matplotlib.dates import YearLocator, MonthLocator
from pandas.plotting import autocorrelation_plot
import seaborn as sns

#DATASET DESCRIPTION
#The data set is at 10 min for about 4.5 months

#ATTRIBUTE INFO
# COL 0 - date time year-month-day hour:minute:second
# COL 1 - Appliances, energy use in Wh
# COL 2 - lights, energy use of light fixtures in the house in Wh


#Load CSV with Pandas
ts = pd.read_csv('energydata_complete.csv', header = 0, sep=',', quotechar='"', engine='python', usecols=[0, 1, 2], dtype = {"date" : "datetime64[ns]", "Appliances" : "float", "lights" : "float"})
#selecting the first three columns of the dataset: "data", “Appliances” and “Lights” 
#which measure the energy consumption of appliances and lights, respectively, across a period of 4.5 months
#load colums separately
appl_data = pd.read_csv('energydata_complete.csv', header = 0, sep=',', quotechar='"', engine='python', usecols=[0, 1], dtype = {"Appliances" : "float"})
lights_data = pd.read_csv('energydata_complete.csv', header = 0, sep=',', quotechar='"', engine='python', usecols=[0, 2], dtype = {"lights" : "float"})

#datetime format
ts.index = pd.to_datetime(ts['date'], format='%Y-%m-%d %H:%M:%S')
appl_data.index = pd.to_datetime(appl_data['date'], format='%Y-%m-%d %H:%M:%S')
lights_data.index = pd.to_datetime(lights_data['date'], format='%Y-%m-%d %H:%M:%S')

#indexes
ts = ts.set_index('date')
appl_data = appl_data.set_index('date')
lights_data = lights_data.set_index('date')

#print data
print(ts)
print(appl_data)
print(lights_data)

#print max and min
print('Max consumption ', lights_data.max())
print('Min consumption ', lights_data.min())
print('Max consumption ', appl_data.max())
print('Min consumption ', appl_data.min())



'''
                                OBSERVED TIMESERIES PLOT
'''
# MONTH

'''
plt.figure(0)
ts['Appliances'].plot(color='b')
plt.title('Appliances - monthly consumption: ')
plt.xlabel('Month')
plt.ylabel('Wh')

plt.figure(1)
ts['lights'].plot(color='y')
plt.title('Lights - monthly consumption: ')
plt.xlabel('Month')
plt.ylabel('Wh')

# ALCUNI MESI HANNO PIU' MISURAZIONI QUINI LA SOMMA DEI CONSUMI NON E' RAPPRESENTATIVA

plt.figure(2)
month = ts['Appliances'].groupby(ts.index.month).sum().plot() 
plt.title('Appliances monthy consumption (total): ')
plt.xlabel('Month')
plt.ylabel('Sum over all Wh')

plt.figure(0)
month = ts['lights'].groupby(ts.index.month).sum().plot() 
plt.title('Lights monthy consumption (total): ')
plt.xlabel('Month')
plt.ylabel('Sum over all Wh')
'''

plt.figure(2)
month = ts['Appliances'].groupby(ts.index.month).mean().plot() 
plt.title('Appliances monthly consumption (mean): ')
plt.xlabel('Month')
plt.ylabel('Mean over all Wh')

plt.figure(3)
month = ts['lights'].groupby(ts.index.month).mean().plot() 
plt.title('Lights monthly consumption (mean): ')
plt.xlabel('Month')
plt.ylabel('Mean over all Wh')

''' BARPLOTS
plt.figure(3)
month = ts['Appliances'].groupby(ts.index.month).sum().plot(kind='bar') 
plt.title('Appliances monthy consumption (total): ')
plt.xlabel('Month')
plt.ylabel('Sum over all Wh')

plt.figure(1)
month = ts['lights'].groupby(ts.index.month).sum().plot(kind='bar') 
plt.title('Lights monthy consumption (total): ')
plt.xlabel('Month')
plt.ylabel('Sum over all Wh')
'''

# DAY
'''
plt.figure(5)
#hour = ts.groupby(ts.index.hour).plot() #RESTITUISCE UN GRAFICO PER OGNI ORA (tot 24 plot)
hour = ts.groupby(ts.index.hour).sum().plot()
plt.title('Daily consumption (total): ')
plt.xlabel('Hour')
plt.ylabel('Sum over all Wh')

plt.figure(6)
hour = ts.groupby(ts.index.hour).sum().plot(kind='bar')
plt.title('Daily consumption (total): ')
plt.xlabel('Hour')
plt.ylabel('Sum over all Wh')
'''

plt.figure(4)
hour = ts['Appliances'].groupby(ts.index.hour).mean().plot()
plt.title('Hourly consumption of appliances (mean): ')
plt.xlabel('Hour')
plt.ylabel('Mean over all Wh')

plt.figure(5)
hour = ts['lights'].groupby(ts.index.hour).mean().plot()
plt.title('Hourly consumption of lights (mean): ')
plt.xlabel('Hour')
plt.ylabel('Mean over all Wh')

plt.figure(6)
hour = ts['Appliances'].groupby(ts.index.hour).mean().plot(kind='bar')
plt.title('Hourly consumption of appliances (mean): ')
plt.xlabel('Hour')
plt.ylabel('Mean over all Wh')

plt.figure(7)
hour = ts['lights'].groupby(ts.index.hour).mean().plot(kind='bar')
plt.title('Hourly consumption of lights (mean): ')
plt.xlabel('Hour')
plt.ylabel('Mean over all Wh')

plt.figure(15)
hour = ts['Appliances'].groupby(ts.index.day).mean().plot(kind='bar')
plt.title('Daily consumption of appliances (mean): ')
plt.xlabel('Day')
plt.ylabel('Mean over all Wh')

plt.figure(16)
hour = ts['lights'].groupby(ts.index.day).mean().plot(kind='bar')
plt.title('Daily consumption of lights (mean): ')
plt.xlabel('Day')
plt.ylabel('Mean over all Wh')

# MONTH - DAY
plt.figure(8)   # evidente regolarità per gli elettrodomestici
                # inferire qualcosa sul consumo di luci (es. ora legale/solare)
hour = ts['Appliances'].groupby([ts.index.month, ts.index.hour]).mean().plot()
plt.title('Daily consumption of appliances for each month (mean): ')
plt.xlabel('Month-Hour')
plt.ylabel('Sum over all Wh')

# MONTH - DAY
plt.figure(9)   # evidente regolarità per gli elettrodomestici
                # inferire qualcosa sul consumo di luci (es. ora legale/solare)
hour = ts['lights'].groupby([ts.index.month, ts.index.hour]).mean().plot()
plt.title('Daily consumption of lights for each month (mean): ')
plt.xlabel('Month-Hour')
plt.ylabel('Sum over all Wh')

'''
                                FREQUENCIES PLOT
'''
# DOMINIO SPETTRALE 

plt.figure(10)
ts['Appliances'].plot(kind='hist', label = 'Appliances')
plt.title('Frequency of appliances consumption: ')
plt.xlabel('Wh')
plt.figure(11)
ts['lights'].plot(kind='hist', label = 'lights')
plt.title('Frequency of appliances consumption: ')
plt.xlabel('Wh')


# inferire info interessanti

'''
                                DENSITY PLOT
 '''
#This is like the histogram, except a function is used to fit the distribution 
#of observations and a smooth line is used to summarize this distribution

plt.figure(12)
ts['Appliances'].plot(kind='kde', c='r')
# In statistics, kernel density estimation (KDE) is a non-parametric way to estimate the probability density function (PDF) of a random variable. 
# This function uses Gaussian kernels and includes automatic bandwidth determination.
plt.title('Density of appliances consumption: ')
plt.xlabel('Wh')
plt.axis([-100,1000,-0.001,0.015])
# distribuzione di probabilità dei watt
# 

plt.figure(13)
ts['lights'].plot(kind='kde', c='r')
plt.title('Density of light consumption: ')
plt.xlabel('Wh')
plt.axis([-10,60,-0.005,0.3])
# inferire info interessanti

'''
                                HEAT MAPS

plt.figure(16)
light_map = lights_data.groupby(ts.index.hour).sum()
sns.heatmap(light_map, annot=False)
plt.title('Sum of lights hourly consumption: ')
plt.ylabel('Hour')

plt.figure(17)
appl_map = appl_data.groupby(ts.index.hour).sum()
sns.heatmap(appl_map, annot=False)
plt.title('Sum of appliances hourly consumption: ')
plt.ylabel('Hour')

plt.figure(18)
light_map2 = lights_data.groupby(ts.index.month).sum()
sns.heatmap(light_map2, annot=False)
plt.title('Sum of lights monthly consumption: ')
plt.ylabel('Month')

plt.figure(19)
appl_map2 = appl_data.groupby(ts.index.month).sum()
sns.heatmap(appl_map2, annot=False)
plt.title('Sum of appliances monthly consumption: ')
plt.ylabel('Month')
'''



'''
                            AUTOCORRELATION PLOT 
'''

'''
plt.figure(14)
autocorrelation_plot(ts['Appliances'])  # --> sinusoidal
plt.title('Appliances autocorrelation: ')

plt.figure(15)
autocorrelation_plot(ts['lights'])      # --> moderate autocorrelation
plt.title('Lights autocorrelation: ')

'''

#Autocorrelation plots are often used for checking randomness in time series. 
#This is done by computing autocorrelations for data values at varying time lags. 
#If time series is random, such autocorrelations should be near zero for any and 
#all time-lag separations. If time series is non-random then one or more of the 
#autocorrelations will be significantly non-zero.

