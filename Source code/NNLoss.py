############################################################
###
### Obtain model loss (mean squared error)
###
############################################################

##### Import Modules #####
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

##### Load Model File #####
model = tf.keras.models.load_model('./model/nn_params_to_iv_3_64.h5')

##### Min Max Scaling #####
X = np.load('./data/params_to_iv_train_params.npy')
y = np.load('./data/params_to_iv_train_iv.npy')

scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

mse = tf.keras.losses.MeanSquaredError()

y_train_pred = model(X_train)
y_test_pred = model(X_test)
loss = mse(y_train_pred, y_train)
val_loss = mse(y_test_pred, y_test)

print("loss = %.10f" % (loss))
print("val_loss = %.10f" % (val_loss))