############################################################
###
### Sample code of option pricing in QuantLib
###
############################################################

##### Code Source: http://gouthamanbalaraman.com/blog/valuing-european-option-heston-model-quantLib.html #####

##### Import Modules #####
import QuantLib as ql
import math

##### Set Current Time #####
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
calculation_date = ql.Date(8, 5, 2015)
ql.Settings.instance().evaluationDate = calculation_date

##### Option Settings #####
maturity_date = ql.Date(8, 5, 2016)
spot_price = 127.62
strike_price = 127
option_type = ql.Option.Call
risk_free_rate = 0.01
dividend_rate = 0.0
volatility = 0.2

##### Set up Quote Handle and Term Structure #####
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
riskfree_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, volatility, day_count))

##### Construct the European Option #####
payoff = ql.PlainVanillaPayoff(option_type, strike_price)
exercise = ql.EuropeanExercise(maturity_date)
european_option = ql.VanillaOption(payoff, exercise)

##### Construct and price BSM process #####
bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, riskfree_ts, flat_vol_ts)
engine = ql.AnalyticEuropeanEngine(bsm_process)
european_option.setPricingEngine(engine)
bsm_price = european_option.NPV()
print ("The BSM model price is ", bsm_price)
bsm_iv = european_option.impliedVolatility(bsm_price, bsm_process)
print ("The BSM model IV is", bsm_iv)

##### Heston parameters #####
v0 = 0.04; kappa = 1; theta = v0; rho = -0.75; sigma = 0.2

##### Construct and price Heston process #####
process = ql.HestonProcess(riskfree_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
model = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model)
european_option.setPricingEngine(engine)
h_price = european_option.NPV()
print ("The Heston model price is",h_price)
h_iv = european_option.impliedVolatility(h_price, bsm_process)
print ("The Heston model IV is",h_iv)