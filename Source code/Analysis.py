############################################################
###
### Result analysis
###
############################################################

import numpy as np
import pandas as pd

calib_params_nn_4_128 = np.array(np.load('./sol/calib_params_nn_4_128.npy'))
calib_params_nn_6_256 = np.array(np.load('./sol/calib_params_nn_6_256.npy'))
calib_params_nn_8_512 = np.array(np.load('./sol/calib_params_nn_8_512.npy'))
calib_params_nn_params_to_iv_3_64 = np.array(np.load('./sol/calib_params_nn_params_to_iv_3_64.npy'))
calib_params_nn_params_to_iv_4_128 = np.array(np.load('./sol/calib_params_nn_params_to_iv_4_128.npy'))
calib_params_ql = np.array(np.load('./sol/calib_params_ql.npy'))
calib_params_ql[:, [0,2,4]] = calib_params_ql[:, [4,0,2]]    # Adjust the order to [v0, kappa, theta, rho, sigma]

calib_errIV_nn_4_128 = np.array(np.load('./sol/calib_errIV_nn_4_128.npy'))
calib_errIV_nn_6_256 = np.array(np.load('./sol/calib_errIV_nn_6_256.npy'))
calib_errIV_nn_8_512 = np.array(np.load('./sol/calib_errIV_nn_8_512.npy'))
calib_errIV_nn_params_to_iv_3_64 = np.array(np.load('./sol/calib_errIV_nn_params_to_iv_3_64.npy'))
calib_errIV_nn_params_to_iv_4_128 = np.array(np.load('./sol/calib_errIV_nn_params_to_iv_4_128.npy'))
calib_errIV_ql = np.array(np.load('./sol/calib_errIV_ql.npy'))

real_params = np.array(np.load('./data/calib_params.npy'))
real_params = real_params[:, 2:7]

ranges = [0.32, 5, 0.32, 0.9, 0.7]

diff_ql = abs(calib_params_ql - real_params) / ranges
diff_nn_4_128 = abs(calib_params_nn_4_128 - real_params) / ranges
diff_nn_6_256 = abs(calib_params_nn_6_256 - real_params) / ranges
diff_nn_8_512 = abs(calib_params_nn_8_512 - real_params) / ranges
diff_nn_params_to_iv_3_64 = abs(calib_params_nn_params_to_iv_3_64 - real_params) / ranges
diff_nn_params_to_iv_4_128 = abs(calib_params_nn_params_to_iv_4_128 - real_params) / ranges

print('nn_4_128 params abs error:')
print(pd.DataFrame(diff_nn_4_128).describe())
print('nn_6_256 params abs error:')
print(pd.DataFrame(diff_nn_6_256).describe())
print('nn_8_512 params abs error:')
print(pd.DataFrame(diff_nn_8_512).describe())
print('nn_params_to_iv_3_64 params abs error:')
print(pd.DataFrame(diff_nn_params_to_iv_3_64).describe())
print('nn_params_to_iv_4_128 params abs error:')
print(pd.DataFrame(diff_nn_params_to_iv_4_128).describe())
print('ql params abs error:')
print(pd.DataFrame(diff_ql).describe())

print('nn_4_128 IV abs %% error:')
print(pd.DataFrame(calib_errIV_nn_4_128).describe())
print('nn_6_256 IV abs %% error:')
print(pd.DataFrame(calib_errIV_nn_6_256).describe())
print('nn_8_512 IV abs %% error:')
print(pd.DataFrame(calib_errIV_nn_8_512).describe())
print('nn_params_to_iv_3_64 IV abs %% error:')
print(pd.DataFrame(calib_errIV_nn_params_to_iv_3_64).describe())
print('nn_params_to_iv_4_128 IV abs %% error:')
print(pd.DataFrame(calib_errIV_nn_params_to_iv_4_128).describe())
print('ql IV abs %% error:')
print(pd.DataFrame(calib_errIV_ql).describe())