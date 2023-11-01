############################################################
###
### NN model building and training
### Model: params-IVS; 4 layers; 128 neurons per layer
###
############################################################

##### Import Modules #####
import math
import pandas as pd
import numpy as np
import tensorflow as tf

from keras import Input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

X = np.load('./data/train_params.npy')
y = np.load('./data/train_ivs.npy')

scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = Sequential()

model.add(Dense(128, input_dim = X_train.shape[1]))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(y_train.shape[1]))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=400, min_lr=9e-10, verbose=1)
checkpoint = ModelCheckpoint('./model/nn_4_128.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

m = model.fit(X_train, y_train, batch_size = 1024, epochs = 5000, verbose = 1, validation_data = (X_test, y_test), callbacks = [checkpoint,reduceLR])

plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

