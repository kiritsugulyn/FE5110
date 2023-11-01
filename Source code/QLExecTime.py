############################################################
###
### Obtain execution time of one IVS prediction
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

##### Set Rates #####
risk_free_rate = X[0,0]
dividend_rate = X[0,1]

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
for k, _ in enumerate(y[0]):
    i = k % len(moneyness)
    j = k // len(moneyness)
    strike = spot_price / moneyness[i]
    tenor = ql.Period(tenors[j])
    vol_quote = ql.QuoteHandle(ql.SimpleQuote(y[0,k]))
    helper = ql.HestonModelHelper(tenor, calendar, spot_price, strike, vol_quote, riskfree_ts, dividend_ts)
    helper.setPricingEngine(engine)
    heston_helpers.append(helper)

start = time.time()
for k in range(1000):
    params_ = ql.Array(list(X[0,2:7]))
    model.setParams(params_)
    error = [h.calibrationError() for h in heston_helpers]
end = time.time()
exec_time = end - start
print("excecution time per prediction = %f" % (exec_time/1000))