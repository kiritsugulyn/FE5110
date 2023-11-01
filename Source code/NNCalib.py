############################################################
###
### Calibration with theoretical IVS data
### Model: params-IVS
###
############################################################

##### Import Modules #####
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import QuantLib as ql
import time

from scipy.optimize import minimize, differential_evolution

##### Load Model File #####
model = tf.keras.models.load_model('./model/nn_4_128.h5')

##### Min Max Scaling #####
X_train_data = np.load('./data/train_params.npy')
y_train_data = np.load('./data/train_ivs.npy')
X_max = np.amax(X_train_data,axis=0)
X_min = np.amin(X_train_data,axis=0)
y_max = np.amax(y_train_data,axis=0)
y_min = np.amin(y_train_data,axis=0)

def min_max_scale(d, d_min, d_max):
    return (d - d_min) / (d_max - d_min)

def min_max_unscale(s, d_min, d_max):
    return d_min + s * (d_max - d_min)

##### Load Calibration Test Data #####
X = np.load('./data/calib_params.npy')
y = np.load('./data/calib_ivs.npy')


##### Calibration Helper Function #####
def cost_fun_nn(y, rates):
    def cost_fun(params):
        x = np.concatenate((rates, params), axis = None)
        pred = model(np.array([x]), training=False).numpy().ravel()  # model.predict() execute very slowly since TF2.1 https://github.com/tensorflow/tensorflow/issues/40261
        diff = pred - y
        return np.sum(abs(diff)/y)
    return cost_fun

def calib_nn(model, y, rates):
    y = min_max_scale(y, y_min, y_max)
    rates = min_max_scale(rates, X_min[0:2], X_max[0:2])
    cost_fun = cost_fun_nn(y, rates)
    start = time.time()
    sol = differential_evolution(func=cost_fun, bounds=[(0,1),(0,1),(0,1),(0,1),(0,1)], maxiter=100)
    end = time.time()
    exec_time = end - start
    X_pred = np.concatenate((rates, sol.x), axis = None)
    y_pred = model(np.array([X_pred]), training=False).numpy().ravel()
    calib_params = min_max_unscale(X_pred, X_min, X_max)[2:7]
    calib_ivs = min_max_unscale(y_pred, y_min, y_max)
    errIV = sol.fun / len(y_pred) * 100
    return calib_params, calib_ivs, errIV, exec_time, sol.nit

##### Calibration using NN #####
calib_params_NN = []
calib_ivs_NN = []
calib_errIV_NN = []

avg_time = 0
avg_nit = 0
avg_errIV = 0

for k in range(X.shape[0]):
    rates = np.array([X[k,0:2]]).ravel()
    real_params = np.array([X[k,2:7]]).ravel()
    calib_params, calib_ivs, errIV, exec_time, nit = calib_nn(model, y[k], rates)

    print("calibration time = %f no. of iteration = %f" % (exec_time, nit))
    print("calib v0 = %f kappa = %f theta = %f rho = %f sigma = %f" % (calib_params[0], calib_params[1], calib_params[2], calib_params[3], calib_params[4]))
    print("real v0 = %f kappa = %f theta = %f rho = %f sigma = %f" % (real_params[0], real_params[1], real_params[2], real_params[3], real_params[4]))
    print("abs IV Error (%%) : %5.3f" % (errIV))

    calib_params_NN.append(calib_params)
    calib_ivs_NN.append(calib_ivs)
    calib_errIV_NN.append(errIV)

    avg_time += exec_time
    avg_nit += nit
    avg_errIV += errIV

np.save('./sol/calib_params_nn_4_128', calib_params_NN)
np.save('./sol/calib_ivs_nn_4_128', calib_ivs_NN)
np.save('./sol/calib_errIV_nn_4_128', calib_errIV_NN)

print("Average Execution Time: %f" %(avg_time/X.shape[0]))
print("Average No. of Iteration: %f" %(avg_nit/X.shape[0]))
print("Average Abs IV Error (%%): %f" %(avg_errIV/X.shape[0]))

