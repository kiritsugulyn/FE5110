############################################################
###
### Obtain execution time of one IVS prediction
### Model: params-IVS
###
############################################################

##### Import Modules #####
import math
import numpy as np
import tensorflow as tf
import time

##### Load Model File #####
model = tf.keras.models.load_model('./model/nn_4_128.h5')

##### Load Calibration Test Data #####
X = np.load('./data/calib_params.npy')
y = np.load('./data/calib_ivs.npy')

start = time.time()
for k in range(100):
    for i in range(X.shape[0]):
        model(np.array([X[i]]), training=False)
end = time.time()
exec_time = end - start
print("excecution time per prediction = %f" % (exec_time/X.shape[0]/100))