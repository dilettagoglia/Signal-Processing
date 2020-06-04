# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:09:06 2020

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

#DATASET DESCRIPTION
# The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy meters. Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets using the date and time column. Two random variables have been included in the data set for testing the regression models and to filter out non predictive attributes (parameters).
# Number of Instances: 19735
# Missing Values? N/A


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
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from keras.utils.data_utils import get_file
from keras.callbacks import LambdaCallback
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import LSTM
from keras.layers import Dropout, Softmax
from keras.optimizers import RMSprop, SGD


''' Model 1) consider only temperature and humidity data as input '''
# multivariate input data
#Load CSV with Pandas
dataset = pd.read_csv('energydata_complete.csv', header = 0, sep=',', quotechar='"', engine='python', usecols=[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18, 19, 20, 1], 
                           dtype = {"date" : "datetime64[ns]", "T1" : "float", "RH_1" : "float", "T2" : "float", "RH_2" : "float", "T3" : "float", "RH_3" : "float", "T4" : "float", "RH_4" : "float", "T5" : "float", "RH_5" : "float", "T6" : "float", "RH_6" : "float", "T7" : "float", "RH_7" : "float", "T8" : "float", "RH_8" : "float", "T9" : "float", "RH_9" : "float", "Appliances" : "float"})

#datetime format
dataset.index = pd.to_datetime(dataset['date'], format='%Y-%m-%d %H:%M:%S')
dataset = dataset.set_index('date')

''' Plot dataset1 '''
values = dataset.values
# specify columns to plot
groups = [0]
i = 1
# plot each column
plt.figure(0)
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
plt.show()

print(dataset)

''' DATASET SPLITTING '''
train = dataset["2016-01-11":"2016-04-30"] # jan - apr
test = dataset["2016-05-11":"2016-05-27"] # may

train.index = pd.to_datetime(train.index, format='%Y-%m-%d %H:%M:%S')
test.index = pd.to_datetime(test.index, format='%Y-%m-%d %H:%M:%S')

'''
#datetime format
train.index = pd.to_datetime(train['date'], format='%Y-%m-%d %H:%M:%S')
test.index = pd.to_datetime(test['date'], format='%Y-%m-%d %H:%M:%S')

#indexes
train = train.set_index('date')
val = val.set_index('date')
test = test.set_index('date')
'''

training_values = train.values
test_values = test.values

#print(training_values)
#print(training_values.shape)

''' INPUT - OUTPUT '''
# split into input and outputs
X_train, y_train = training_values[:, 1:], training_values[:, :1] 
# temperature e umidità (9+9 tot. 18 feature) sono input mentre la prima colonna (appliances) è l'output
X_test, y_test = test_values[:, 1:], test_values[:, :1]

''' PREPROCESSING '''
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit_transform(X_train)
X_train = scaler.transform(X_train)
scaler.fit_transform(y_train)
y_train = scaler.transform(y_train)
scaler.fit_transform(X_test)
X_test = scaler.transform(X_test)
scaler.fit_transform(y_test)
y_test = scaler.transform(y_test)

# just a check
print('\n\nBefore reshaping:')
print('\nTrain_input (temperature and humidity)\n', X_train[0:5], '\nTrain_output (energy)\n', y_train[0:5], 
      '\nTest_input (temperature and humidity)\n', X_test[0:5], '\nTest_output (energy)\n', y_test[0:5])
print('\nTrain_input (temperature and humidity)\n', X_train.shape, '\nTrain_output (energy)\n', y_train.shape, 
      '\nTest_input (temperature and humidity)\n', X_test.shape, '\nTest_output (energy)\n', y_test.shape)



print('\nBefore reshaping:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print('\nAfter reshaping:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)



''' HYPERPARAM '''
learning_rate=1e-4
batch_size=70
epochs=20


''' BUILDING THE FIRST MODEL '''
# Ora costriuiamo il modello Keras LSTM utilizzando il modulo Sequential e, 
# visto che andremo a utilizzare una GPU per velocizzare il processo di apprendimento 
# utilizzeremo il layer CuDNNLSTM fornito da Keras ottimizzato per essere eseguito su runtime GPU.

print('\n\nCompilo il modello...')
# using Sequential API
model = tf.keras.Sequential()   # instaciate a model using Sequential class 
                                # --> will contruct a pipeline of layers
# add one layer at time
                               

model.add(layers.LSTM(30, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))
model.compile(loss='mae', optimizer='adam')

''' MODEL FIT '''
# fit model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), 
                     verbose=1, shuffle=False)

model.summary()


# Plot Model Loss 
# list all data in history
print(history.history.keys())

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

# make predictions

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

print(train_predict.shape, test_predict.shape)

# invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)


#train_predict = train_predict0.reshape((train_predict0.shape[0], train_predict0.shape[2]))
#test_predict = test_predict0.reshape((test_predict0.shape[0], test_predict0.shape[2]))


# invert predictions
#train_predict = scaler.inverse_transform(train_predict)
#test_predict = scaler.inverse_transform(test_predict)

print(train_predict[:,0])

aa=[x for x in range(2000)]
plt.figure(figsize=(9,5))
plt.plot(aa, y_train[:, 0][:2000], marker='.', label="actual")
plt.plot(aa, train_predict[:,0][:2000], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();
aa=[x for x in range(2000)]
plt.figure(figsize=(8,4))
plt.plot(aa, y_test[:, 0][:2000], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:2000], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

pred = pd.DataFrame({'Predicted': test_predict[:, 0]}, index=test.index)

plt.figure(9)
v = test['Appliances'].groupby(test.index.hour).mean()
z = pred.groupby(pred.index.hour).mean()
plt.plot(v, c='r')
plt.plot(z, c='b')