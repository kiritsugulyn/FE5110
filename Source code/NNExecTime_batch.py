############################################################
###
### Obtain execution time of one IVS prediction
### Model: params-IV
###
############################################################

##### Import Modules #####
import math
import numpy as np
import tensorflow as tf
import time
import QuantLib as ql

##### Load Model File #####
model = tf.keras.models.load_model('./model/nn_params_to_iv_4_128.h5')

##### Load Calibration Test Data #####
X = np.load('./data/calib_params.npy')
y = np.load('./data/calib_ivs.npy')

##### Min Max Scaling #####
X_train_data = np.load('./data/params_to_iv_train_params.npy')
y_train_data = np.load('./data/params_to_iv_train_iv.npy')
X_max = np.amax(X_train_data,axis=0)
X_min = np.amin(X_train_data,axis=0)
y_max = np.amax(y_train_data,axis=0)
y_min = np.amin(y_train_data,axis=0)

def min_max_scale(d, d_min, d_max):
    return (d - d_min) / (d_max - d_min)

def min_max_unscale(s, d_min, d_max):
    return d_min + s * (d_max - d_min)

##### Create Vol Surface #####
moneyness = [.8, .85, .9, .95, .975, .99, 1.0, 1.01, 1.025, 1.05, 1.1, 1.15, 1.2] # S_0/K 
tenors = ['1W', '2W', '1M', '2M', '3M', '6M', '1Y'] # T
calculation_date = ql.Date(8, 2, 2019)
ttm_days = [calculation_date + ql.Period(tenor) - calculation_date for tenor in tenors]
vol_surface = []
for t in ttm_days:
    for m in moneyness:
        vol_surface.append([m, t])
vol_surface = min_max_scale(vol_surface, X_min[2:4], X_max[2:4])
vol_surface = np.array(vol_surface)

start = time.time()
for k in range(100):
    for i in range(X.shape[0]):
        model(np.concatenate((np.tile(X[i,0:2], (vol_surface.shape[0],1)), vol_surface, np.tile(X[i,2:7], (vol_surface.shape[0],1))), axis=1), training=False)
end = time.time()
exec_time = end - start
print("excecution time per prediction = %f" % (exec_time/X.shape[0]/100))