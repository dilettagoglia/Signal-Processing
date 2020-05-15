# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:28:33 2020

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
import cv2
import scipy
from statistics import variance 
import matplotlib.cm as cm


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
#print(ts)
#print(appl_data)
#print(lights_data)

#print max and min
print('Max consumption ', lights_data.max())
print('Min consumption ', lights_data.min())
print('Max consumption ', appl_data.max())
print('Min consumption ', appl_data.min())


''' 
                                HIDDEN MARKOV MODEL 
'''
#number of hidden states
ncomp = 2 # then try also with 3, 5, 7
# Create an HMM with Gaussian emission for 'Appliances' 
appl_hmm = GaussianHMM(n_components=ncomp , covariance_type="diag",
                       n_iter=1000, algorithm='viterbi') # model for appliances
# Create an HMM with Gaussian emission for 'light' 
lights_hmm = GaussianHMM(n_components=ncomp , covariance_type="diag",
                       n_iter=1000, algorithm='viterbi') # model for lights
# fit models to data
appl_hmm.fit(appl_data) 
lights_hmm.fit(lights_data)

# creating a subsequence to perform Viterbi (1 month)
# APPL
subseq = appl_data['2016-03':'2016-04'] # tutto il mese di marzo
subseq.index = pd.to_datetime(subseq.index, format='%Y-%m-%d %H:%M:%S')
# LIGHTS
subseq2 = lights_data['2016-03':'2016-04'] # tutto il mese di marzo
subseq2.index = pd.to_datetime(subseq2.index, format='%Y-%m-%d %H:%M:%S')
#print(subseq)

# Viterbi retrieves the most probable path through a HMM that generates a certain observation X
# Decode the optimal sequence of internal hidden state (Viterbi)
hidden_states = appl_hmm.predict(subseq)# Predict the hidden states of HMM
hidden_states2 = lights_hmm.predict(subseq2)
#print('Hidden states:', hidden_states)
#print('Total hidden states assigned:', len(hidden_states))

'''
#plotting the predicted subsequence 
plt.figure(0)
subseq.plot(c='lightgrey', zorder=-1)
plt.title('Appliances consumption of March (subsequence): ')
plt.xlabel('Day')
plt.ylabel('Sum of consumptions (Wh)')
'''

'''
                                    VITERBI PLOTS
'''

'''
                                    2 HS
'''
'''
# APPL
plt.figure(0)
plt.title('Timeseries of '+ str(ncomp) + 'hidden states for appliances (Viterbi): ')

for i in range(len(hidden_states)):
    if hidden_states[i] == 1:
        plt.plot(subseq.index[i], subseq['Appliances'][i], c='r', label='High')
    else:
        plt.plot(subseq.index[i], subseq['Appliances'][i], c='g', label='Low')
# LIGHTS
plt.figure(1)
plt.title(str(ncomp) + 'hidden states for lights: ')

for i in range(len(hidden_states2)):
    if hidden_states2[i] == 1:
        plt.scatter(subseq2.index[i], subseq2['lights'][i], c='r', label='High')
    else:
        plt.scatter(subseq2.index[i], subseq2['lights'][i], c='g', label='Low')
'''



'''
                                    3 HS
'''
'''
# APPL
plt.figure(0)
plt.title(str(ncomp) + 'hidden states for appliances (Viterbi): ')
for i in range(len(hidden_states)):
    if hidden_states[i] == 1:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='r', label='High')
    if hidden_states[i] == 2:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='y', label='Medium')
    if hidden_states[i] == 0:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='g', label='Low')
# LIGHTS
plt.figure(1)
plt.title(str(ncomp) + 'hidden states for lights (Viterbi): ')
for i in range(len(hidden_states2)):
    if hidden_states2[i] == 1:
        plt.scatter(subseq2.index[i], subseq2['lights'][i], c='r', label='High')
    if hidden_states2[i] == 2:
        plt.scatter(subseq2.index[i], subseq2['lights'][i], c='y', label='Medium')
    if hidden_states2[i] == 0:
        plt.scatter(subseq2.index[i], subseq2['lights'][i], c='g', label='Low')
'''

'''
                                7 HS
'''
'''
# AGGIUNGERE LE RETTE ORIZZONATI DELL EMEDIE PER CAPIRE A QUALE REGIME
# CORRISPONDE CIASCUN HIDDE STATE E COLORARLO DI CONSEGUENZA
plt.figure(0)
plt.plot(subseq, c='yellow')


colors = np.random.rand(7)
states = (pd.DataFrame(hidden_states, columns=['states'], index = subseq.index))
plt.scatter(subseq.index, states, c=hidden_states, alpha=0.5)
'''
# APPL
'''
plt.figure(0, figsize=(15, 10))
plt.title(str(ncomp) + 'hidden states for appliances (Viterbi): ')
for i in range(len(hidden_states)):
    if hidden_states[i] == 1:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='darkred')
    if hidden_states[i] == 2:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='y')
    if hidden_states[i] == 0:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='lightgreen')
    if hidden_states[i] == 3:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='g')
    if hidden_states[i] == 4:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='r')
    if hidden_states[i] == 5:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='blue')
    if hidden_states[i] == 6:
        plt.scatter(subseq.index[i], subseq['Appliances'][i], c='orange')
plt.hlines((38.16, 72.5, 53.44, 97.62, 125.36, 258.14, 502.71), subseq.index[0], subseq.index[4463], linestyles='dotted')
plt.plot(subseq, c='lightgrey', zorder=-1)
'''

'''
                                    STATISTICS
'''
'''
# Indicate the component numbers and mean and var of each component
# APPL
print("\n\nStatistics of appliances model (means and vars of each hidden state) : ")
for i in range(appl_hmm.n_components):
    print("\nHidden state: ", i+1)
    print("mean = ", round(appl_hmm.means_[i][0], 2))
    print("var = ", round(np.diag(appl_hmm.covars_[i])[0], 2))
# LIGHTS
print("\n\nStatistics of lights model (means and vars of each hidden state) : ")
for i in range(lights_hmm.n_components):
    print("\nHidden state: ", i+1)
    print("mean = ", round(lights_hmm.means_[i][0], 2))
    print("var = ", round(np.diag(lights_hmm.covars_[i])[0], 2))
'''



'''
                                MODEL PARAMETERS
'''

# APPL
print('\n\nParameters of appliances model.\n- Transition matrix: \n', appl_hmm.transmat_)
logProb = appl_hmm.score(appl_data)
print('\n Log likelihood: \n', round(logProb,2))
# LIGHTS
print('\n\nParameters of lights model.\n- Transition matrix: \n', lights_hmm.transmat_)
logProb = lights_hmm.score(np.reshape(appl_data,[len(lights_data),1]))
print('\n- Log likelihood: \n', round(logProb,2))




'''
                                    SAMPLING
'''
# Generate new samples (visible, hidden)

# APPL
X1, Z1 = appl_hmm.sample(143) # 143 is the number of measurement per day (24 hours)
plt.figure(2)
plt.plot(X1)
plt.plot(Z1*10)
plt.title('Samples generated for appliances')
plt.xlabel('Samples')
# LIGHTS
X2, Z2 = lights_hmm.sample(143)
plt.figure(3)
plt.plot(X2)
plt.plot(Z2)
plt.title('Samples generated for lights')
plt.xlabel('Samples')

'''
                                PROB DISTRIB COMPARISON
'''

# APPL
plt.figure(4)
X1 = pd.DataFrame(X1, columns=['appl_generated'])
#print(X1)
X1['appl_generated'].plot(kind='kde', label='Generated sample')
plt.axis([-100,1000,-0.001,0.015]) #rescaling x axes
plt.title('Probability distributions of generated samples (appliances)')
legend = plt.legend()


# LIGHTS
plt.figure(5)
X2 = pd.DataFrame(X2, columns=['lights_generated'])
#print(X1)
X2['lights_generated'].plot(kind='kde', label='Generated sample')
plt.axis([-10,60,-0.005,0.3]) #rescaling x axes
plt.title('Probability distributions of generated samples (lights)')
legend = plt.legend()


