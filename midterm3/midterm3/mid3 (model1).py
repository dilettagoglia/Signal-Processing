# -*- coding: utf-8 -*-
"""
Created on Fri May 29 22:56:25 2020

@author: Diletta
"""


''' DATASET WEBSITE: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction '''
''' GITHUB REPOS: https://github.com/LuisM78/Appliances-energy-prediction-data '''


'''                                     ASSIGNMENT 3

- Train a gated recurrent neural network of your choice (LSTM, GRU) to predict energy expenditure (“Appliances” column) using two approaches:

1. Predict the current energy expenditure given as input information the temperature (T_i) and humidity (RH_i) information from all the i sensors in the house.
2. Setup a one step-ahead predictor for energy expenditure, i.e. given the current energy consumption, predict its next value.

- Show and compare performance of both methods. Remember that testing in a one-step ahead prediction does not use teacher forcing.
'''

'''
DATASET DESCRIPTION
The data set is at 10 min for about 4.5 months. The house temperature and humidity 
conditions were monitored with a ZigBee wireless sensor network. 
Each wireless node transmitted the temperature and humidity conditions around 3.3 min. 
Then, the wireless data was averaged for 10 minutes periods. The energy data was logged 
every 10 minutes with m-bus energy meters. Weather from the nearest airport weather 
station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable 
Prognosis (rp5.ru), and merged together with the experimental data sets using the date 
and time column. Two random variables have been included in the data set for testing 
the regression models and to filter out non predictive attributes (parameters).

# Number of Instances: 19735
# Missing Values? N/A

'''

#ATTRIBUTE INFO
# COL 0 - date time year-month-day hour:minute:second
# COL 1 - Appliances, energy use in Wh
# COL 2 - lights, energy use of light fixtures in the house in Wh
# T1, Temperature in kitchen area, in Celsius
# RH_1, Humidity in kitchen area, in %
# T2, Temperature in living room area, in Celsius
# RH_2, Humidity in living room area, in %
# T3, Temperature in laundry room area
# RH_3, Humidity in laundry room area, in %
# T4, Temperature in office room, in Celsius
# RH_4, Humidity in office room, in %
# T5, Temperature in bathroom, in Celsius
# RH_5, Humidity in bathroom, in %
# T6, Temperature outside the building (north side), in Celsius
# RH_6, Humidity outside the building (north side), in %
# T7, Temperature in ironing room , in Celsius
# RH_7, Humidity in ironing room, in %
# T8, Temperature in teenager room 2, in Celsius
# RH_8, Humidity in teenager room 2, in %
# T9, Temperature in parents room, in Celsius
# RH_9, Humidity in parents room, in %

from math import sqrt

import numpy as np
from numpy import concatenate
from numpy import transpose
import pandas as pd
from pandas.plotting import autocorrelation_plot

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import LSTM
from keras.layers import Dropout, Softmax
from keras.optimizers import RMSprop, SGD


''' Model 1) consider temperature and humidity data as input '''
# multivariate input data
#Load CSV with Pandas
dataset = pd.read_csv('energydata_complete.csv', header = 0, sep=',', quotechar='"', engine='python', usecols=[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18, 19, 20, 1], dtype = {"date" : "datetime64[ns]", "T1" : "float", "RH_1" : "float", "T2" : "float", "RH_2" : "float", "T3" : "float", "RH_3" : "float", "T4" : "float", "RH_4" : "float", "T5" : "float", "RH_5" : "float", "T6" : "float", "RH_6" : "float", "T7" : "float", "RH_7" : "float", "T8" : "float", "RH_8" : "float", "T9" : "float", "RH_9" : "float", "Appliances" : "float"})

#datetime format
dataset.index = pd.to_datetime(dataset['date'], format='%Y-%m-%d %H:%M:%S')
dataset = dataset.set_index('date')

# Plot dataset1
values = dataset.values
groups = [0] # specify columns to plot
i = 1
# plot each column
plt.figure(0)
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
plt.show()

''' DATASET SPLITTING '''

train = dataset["2016-01-11":"2016-04-30"] # jan - apr
test = dataset["2016-05-11":"2016-05-27"] # may

train.index = pd.to_datetime(train.index, format='%Y-%m-%d %H:%M:%S')
test.index = pd.to_datetime(test.index, format='%Y-%m-%d %H:%M:%S')

training_values = train.values
test_values = test.values

#print(training_values)
#print(training_values.shape)


''' INPUT - OUTPUT '''
# split into input and outputs
train_X, train_y = training_values[:, 1:], training_values[:, :1] 
# temperature e umidità (9+9 tot. 18 feature) sono input mentre la prima colonna (appliances) è l'output
test_X, test_y = test_values[:, 1:], test_values[:, :1]


''' PREPROCESSING '''
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(train_X)
train_X = scaler.transform(train_X)
scaler.fit_transform(train_y)
train_y = scaler.transform(train_y)
scaler.fit_transform(test_X)
test_X = scaler.transform(test_X)
scaler.fit_transform(test_y)
test_y = scaler.transform(test_y)

# just a check
print('\n\nBefore reshaping:')
print('\nTrain_input (temperature and humidity)\n', train_X[0:5], '\nTrain_output (energy)\n', train_y[0:5], 
      '\nTest_input (temperature and humidity)\n', test_X[0:5], '\nTest_output (energy)\n', test_y[0:5])
print('\nTrain_input (temperature and humidity)\n', train_X.shape, '\nTrain_output (energy)\n', train_y.shape, 
      '\nTest_input (temperature and humidity)\n', test_X.shape, '\nTest_output (energy)\n', test_y.shape)

plt.figure(1)
plt.plot(train_X[:, :1])
plt.title("Appliances (train)")

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

print('\n\nAfter reshaping:')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


''' HYPERPARAM '''
learning_rate=1e-4
batch_size=70
epochs=20


''' BUILDING THE FIRST MODEL '''
# using Sequential API
model = tf.keras.Sequential()   # instaciate a model using Sequential class 
                                # --> will contruct a pipeline of layers
# building add one layer at time               
model.add(layers.LSTM(18, activation='tanh', return_sequences=True, input_dim=(train_X.shape[2])))
# set the return_sequences to True, the output shape becomes a 3D array
model.add(layers.Dropout(0.5)) 
# Dropout regularize the model by ramdomly tuting off some neurons --> prevent overfitting
model.add(layers.Dense(1))
# then you don't need to specify the input shape again because
# it is automatically inferred by sequential layer
#optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(optimizer='adam', loss='mae')


#model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print('\n', model.summary())


''' MODEL FIT '''
# fit model
history = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False)

# Plot Model Loss and Model accuracy
# list all data in history
print(history.history.keys())
'''
# summarize history for accuracy
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])  # RAISE ERROR
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
# summarize history for loss
plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) #RAISE ERROR
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

''' MODEL PREDICTION '''
# make a prediction
test_predict0 = model.predict(test_X)
#reshape (because return_sequences was set to True)
test_predict = test_predict0.reshape((test_predict0.shape[0], 
                                      test_predict0.shape[2]))
# invert predictions
test_predict = scaler.inverse_transform(test_predict)


pred = pd.DataFrame({'Predicted': test_predict[:, 0]}, index=test.index)


print('\nPredicted values:\n', pred)
'''
plt.figure(4)
dataset['Appliances'].plot()
plt.title('Appliances all dataset (training + test)')


plt.figure(5)
test.plot()
plt.title('Test data (appliances, temperatures and humidities)')

#print(test['Appliances'])

plt.figure(6)
plt.plot(test['Appliances']/10)
plt.title('Appliances test data (may)')

plt.figure(7)
pred.plot(c='r')
plt.title('Predicted data for appliances')
'''

'''
plt.figure(8)                                   # DA SISTEMARE !!!
frames = [train['Appliances']/10, pred]
result = pd.concat(frames)
result.plot()
'''

plt.figure(9)
pred.groupby(pred.index.hour).mean().plot(c='r')
plt.show()

plt.figure(10)
v = test['Appliances'].groupby(test.index.hour).mean()
plt.plot(v)
plt.show()

# Autocorrelation plot
plt.figure(11)
autocorrelation_plot(dataset['Appliances']) # --> sinusoidal
plt.title('Appliances autocorrelation: ')

plt.figure(12)
a = dataset.corr()
a.plot()
print(a)

#Autocorrelation plots are often used for checking randomness in time series. 
#This is done by computing autocorrelations for data values at varying time lags. 
#If time series is random, such autocorrelations should be near zero for any and 
#all time-lag separations. If time series is non-random then one or more of the 
#autocorrelations will be significantly non-zero.






''' Appunti lezione 19

model.add(layers.Dense(20, activation='relu', input_shape=(20, ))) 


model.add(layers.Dense(20, activation='relu'))


model.add(layers.Dense(10, activation='softmax'))
# ex. if we want to classify digits into 10 different classes --> layer with output 10

'''