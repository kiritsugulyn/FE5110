############################################################
###
### Calibration with real market data
### Stock: AAPL
### Model: params-IV and traditional QL helper
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

##### Load AAPL IVS Data #####
y = np.load('./aapl/20210226_ivs.npy').ravel()
print("no. of market observations = %d" % (np.sum(y > 0)))

##### Rate Setting #####
risk_free_rate = 0.01407  ## US10Y close 1.2@20210212 1.345@20210219  1.407@20210226
dividend_rate = 0.82 / 121.26
rates = np.array([risk_free_rate, dividend_rate])

############################################################
###################### NN Calibration ######################
############################################################

##### Load Model File #####
model = tf.keras.models.load_model('./model/nn_params_to_iv_4_128.h5')

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
calculation_date = ql.Date(26, 2, 2021)
moneyness = [.8, .85, .9, .95, .975, .99, 1.0, 1.01, 1.025, 1.05, 1.1, 1.15, 1.2] # S_0/K 
tenors = ['1W', '2W', '1M', '2M', '3M', '6M', '1Y'] # T
ttm_days = [calculation_date + ql.Period(tenor) - calculation_date for tenor in tenors]
vol_surface = []
k = 0
for t in ttm_days:
    for m in moneyness:
        if (y[k] > 0):
            vol_surface.append([m, t])
        k += 1
vol_surface = min_max_scale(vol_surface, X_min[2:4], X_max[2:4])
vol_surface = np.array(vol_surface)

##### Calibration Helper Function #####
def cost_fun_nn(y, rates):
    def cost_fun(params):
        x = np.concatenate((rates, vol_surface, np.tile(params, (vol_surface.shape[0], 1))), axis = 1)
        pred = model(x, training=False).numpy().ravel()  # model.predict() execute very slowly since TF2.1 https://github.com/tensorflow/tensorflow/issues/40261
        diff = pred - y
        return np.sum(abs(diff)/y)
    return cost_fun

def calib_nn(model, y, rates):
    y = min_max_scale(y, y_min, y_max)
    rates = min_max_scale(rates, X_min[0:2], X_max[0:2])
    rates = np.tile(rates, (vol_surface.shape[0],1))
    cost_fun = cost_fun_nn(y[y>0], rates)
    start = time.time()
    sol = differential_evolution(func=cost_fun, bounds=[(0,1),(0,1),(0,1),(0,1),(0,1)], maxiter=100)
    end = time.time()
    exec_time = end - start
    X_pred = np.concatenate((rates, vol_surface, np.tile(sol.x, (vol_surface.shape[0], 1))), axis = 1)
    y_pred = model(X_pred, training=False).numpy().ravel()
    calib_params = min_max_unscale(X_pred, X_min, X_max)[0,4:9]
    calib_ivs = min_max_unscale(y_pred, y_min, y_max)
    errIV = sol.fun / np.sum(y > 0) * 100
    return calib_params, calib_ivs, errIV, exec_time, sol.nit

##### Calibration function call #####
print("Using NN model to calibrate.....")
calib_params, calib_ivs, errIV, exec_time, nit = calib_nn(model, y, rates)

print("calibration time = %f no. of iteration = %f" % (exec_time, nit))
print("calib v0 = %f kappa = %f theta = %f rho = %f sigma = %f" % (calib_params[0], calib_params[1], calib_params[2], calib_params[3], calib_params[4]))
print("abs IV Error (%%) : %5.3f" % (errIV))


############################################################
###################### QL Calibration ######################
############################################################

#### Set Current Time #####
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
calculation_date = ql.Date(26, 2, 2021)
ql.Settings.instance().evaluationDate = calculation_date

##### Set Base Condition #####
spot_price = 121.26   ## AAPL close 135.37@20210212 129.87@20210219  121.26@20210226
volatility = 0.2  #dummy
moneyness = [.8, .85, .9, .95, .975, .99, 1.0, 1.01, 1.025, 1.05, 1.1, 1.15, 1.2] # S_0/K 
tenors = ['1W', '2W', '1M', '2M', '3M', '6M', '1Y'] # T

##### Calibration Helper Function #####
def cost_fun_ql(model, helpers, norm=False):
    def cost_fun(params):
        params_ = ql.Array(list(params))
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers]
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error
    return cost_fun

def calib_ql(X, y):
    ##### Set up Quote Handle and Term Structure #####
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    riskfree_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
    flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, volatility, day_count))

    ##### Build Heston Model #####
    v0 = 0.1; kappa = 1; theta = 0.1; rho = -0.75; sigma = 0.2 # dummy parameters
    process = ql.HestonProcess(riskfree_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    ##### Build Heston Helpers #####
    heston_helpers = []
    for k, _ in enumerate(y):
        if y[k] > 0:
            i = k % len(moneyness)
            j = k // len(moneyness)
            strike = spot_price / moneyness[i]
            tenor = ql.Period(tenors[j])
            vol_quote = ql.QuoteHandle(ql.SimpleQuote(y[k]))
            helper = ql.HestonModelHelper(tenor, calendar, spot_price, strike, vol_quote, riskfree_ts, dividend_ts)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)

    ##### Model Calibration #####
    cost_fun = cost_fun_ql(model, heston_helpers, norm=True)
    start = time.time()
    sol = differential_evolution(cost_fun, bounds=[(0.04,0.36),(0,5),(0.1,0.8),(-0.9,0),(0.04,0.36)], maxiter=100)
    end = time.time()
    theta, kappa, sigma, rho, v0 = model.params()
    exec_time = end - start
    print("calibration Time = %f no. of iteration = %f" % (exec_time, sol.nit))
    print("calib v0 = %f kappa = %f theta = %f rho = %f sigma = %f" % (v0, kappa, theta, rho, sigma))

    ##### Calculate BSM IV #####
    def heston_iv (strike, tenor):
        ##### Construct European option #####
        option_type = ql.Option.Call
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = ql.EuropeanExercise(calculation_date+ql.Period(tenor))
        european_option = ql.VanillaOption(payoff, exercise)

        ##### Construct and price BSM process #####
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, riskfree_ts, flat_vol_ts)
        engine = ql.AnalyticHestonEngine(model)
        european_option.setPricingEngine(engine)
        bsm_iv = european_option.impliedVolatility(european_option.NPV(), bsm_process)
    
        return bsm_iv
    
    ##### Model Validation #####
    errIV = 0.0
    calib_ivs = []
    for k, _ in enumerate(y):
        if y[k] > 0:
            i = k % len(moneyness)
            j = k // len(moneyness)
            strike = spot_price / moneyness[i]
            tenor = tenors[j]
            h_iv = heston_iv (strike, tenor)
            calib_ivs.append(h_iv)
            errIV_k = (h_iv/y[k] - 1.0)
            errIV += abs(errIV_k)
        else:
            calib_ivs.append(0)
    errIV = errIV*100.0/np.sum(y > 0)
    print ("abs IV Error (%%) : %5.3f" % (errIV))

    return [theta, kappa, sigma, rho, v0], calib_ivs, errIV, exec_time, sol.nit


##### Calibration function call #####
print("Using QL helper to calibrate.....")
calib_params, calib_ivs, errIV, exec_time, nit = calib_ql(rates, y)