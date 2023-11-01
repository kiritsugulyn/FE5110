############################################################
###
### Calibration with theoretical IVS data
### Model: QL helper
###
############################################################

##### Import Modules #####
import math
import pandas as pd
import numpy as np
import QuantLib as ql
import time

from scipy.optimize import minimize, differential_evolution

##### Load Calibration Test Data #####
X = np.load('./data/calib_params.npy')
y = np.load('./data/calib_ivs.npy')

#### Set Current Time #####
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
calculation_date = ql.Date(8, 2, 2019)
ql.Settings.instance().evaluationDate = calculation_date

##### Set Base Condition #####
spot_price = 100
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
    ##### Set Rates #####
    risk_free_rate = X[0]
    dividend_rate = X[1]

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
    print("real v0 = %f kappa = %f theta = %f rho = %f sigma = %f" % (X[2], X[3], X[4], X[5], X[6]))

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
    for k, _ in enumerate(heston_helpers):
        i = k % len(moneyness)
        j = k // len(moneyness)
        strike = spot_price / moneyness[i]
        tenor = tenors[j]
        try:
            h_iv = heston_iv (strike, tenor)
        except RuntimeError as e:
            print(k,e)
            calib_ivs.append(0)
        else:
            calib_ivs.append(h_iv)
            errIV_k = (h_iv/y[k] - 1.0)
            errIV += abs(errIV_k)
    errIV = errIV*100.0/np.sum(np.array(calib_ivs) > 0)
    print ("abs IV Error (%%) : %5.3f" % (errIV))

    return [theta, kappa, sigma, rho, v0], calib_ivs, errIV, exec_time, sol.nit


##### Calibration Loop #####
calib_params_ql = []
calib_ivs_ql = []
calib_errIV_ql = []

avg_time = 0
avg_nit = 0
avg_errIV = 0

for k in range(X.shape[0]):
    calib_params, calib_ivs, errIV, exec_time, nit = calib_ql(X[k], y[k])

    calib_params_ql.append(calib_params)
    calib_ivs_ql.append(calib_ivs)
    calib_errIV_ql.append(errIV)

    avg_time += exec_time
    avg_nit += nit
    avg_errIV += errIV

np.save('./sol/calib_params_ql', calib_params_ql)
np.save('./sol/calib_ivs_ql', calib_ivs_ql)
np.save('./sol/calib_errIV_ql', calib_errIV_ql)

print("Average Execution Time: %f" %(avg_time/X.shape[0]))
print("Average No. of Iteration: %f" %(avg_nit/X.shape[0]))
print("Average Abs IV Error (%%): %f" %(avg_errIV/X.shape[0]))