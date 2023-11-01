############################################################
###
### Sample code of model calibration in QuantLib
###
############################################################

##### Code Source: http://gouthamanbalaraman.com/blog/volatility-smile-heston-model-calibration-quantlib-python.html #####

##### Import Modules #####
import numpy as np
import QuantLib as ql
import math

##### Set Current Time #####
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
calculation_date = ql.Date(6, 11, 2015)
ql.Settings.instance().evaluationDate = calculation_date

##### Set Market Condition #####
spot_price = 659.37
risk_free_rate = 0.01
dividend_rate = 0.0
volatility = 0.2  #dummy

##### Set up Quote Handle and Term Structure #####
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
riskfree_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, volatility, day_count))

##### Input Data #####
expiration_dates = [ql.Date(6,12,2015), ql.Date(6,1,2016), ql.Date(6,2,2016),
                    ql.Date(6,3,2016), ql.Date(6,4,2016), ql.Date(6,5,2016), 
                    ql.Date(6,6,2016), ql.Date(6,7,2016), ql.Date(6,8,2016),
                    ql.Date(6,9,2016), ql.Date(6,10,2016), ql.Date(6,11,2016), 
                    ql.Date(6,12,2016), ql.Date(6,1,2017), ql.Date(6,2,2017),
                    ql.Date(6,3,2017), ql.Date(6,4,2017), ql.Date(6,5,2017), 
                    ql.Date(6,6,2017), ql.Date(6,7,2017), ql.Date(6,8,2017),
                    ql.Date(6,9,2017), ql.Date(6,10,2017), ql.Date(6,11,2017)]
strikes = [527.50, 560.46, 593.43, 626.40, 659.37, 692.34, 725.31, 758.28]
implied_vols = [
[0.37819, 0.34177, 0.30394, 0.27832, 0.26453, 0.25916, 0.25941, 0.26127],
[0.3445, 0.31769, 0.2933, 0.27614, 0.26575, 0.25729, 0.25228, 0.25202],
[0.37419, 0.35372, 0.33729, 0.32492, 0.31601, 0.30883, 0.30036, 0.29568],
[0.37498, 0.35847, 0.34475, 0.33399, 0.32715, 0.31943, 0.31098, 0.30506],
[0.35941, 0.34516, 0.33296, 0.32275, 0.31867, 0.30969, 0.30239, 0.29631],
[0.35521, 0.34242, 0.33154, 0.3219, 0.31948, 0.31096, 0.30424, 0.2984],
[0.35442, 0.34267, 0.33288, 0.32374, 0.32245, 0.31474, 0.30838, 0.30283],
[0.35384, 0.34286, 0.33386, 0.32507, 0.3246, 0.31745, 0.31135, 0.306],
[0.35338, 0.343, 0.33464, 0.32614, 0.3263, 0.31961, 0.31371, 0.30852],
[0.35301, 0.34312, 0.33526, 0.32698, 0.32766, 0.32132, 0.31558, 0.31052],
[0.35272, 0.34322, 0.33574, 0.32765, 0.32873, 0.32267, 0.31705, 0.31209],
[0.35246, 0.3433, 0.33617, 0.32822, 0.32965, 0.32383, 0.31831, 0.31344],
[0.35226, 0.34336, 0.33651, 0.32869, 0.3304, 0.32477, 0.31934, 0.31453],
[0.35207, 0.34342, 0.33681, 0.32911, 0.33106, 0.32561, 0.32025, 0.3155],
[0.35171, 0.34327, 0.33679, 0.32931, 0.3319, 0.32665, 0.32139, 0.31675],
[0.35128, 0.343, 0.33658, 0.32937, 0.33276, 0.32769, 0.32255, 0.31802],
[0.35086, 0.34274, 0.33637, 0.32943, 0.3336, 0.32872, 0.32368, 0.31927],
[0.35049, 0.34252, 0.33618, 0.32948, 0.33432, 0.32959, 0.32465, 0.32034],
[0.35016, 0.34231, 0.33602, 0.32953, 0.33498, 0.3304, 0.32554, 0.32132],
[0.34986, 0.34213, 0.33587, 0.32957, 0.33556, 0.3311, 0.32631, 0.32217],
[0.34959, 0.34196, 0.33573, 0.32961, 0.3361, 0.33176, 0.32704, 0.32296],
[0.34934, 0.34181, 0.33561, 0.32964, 0.33658, 0.33235, 0.32769, 0.32368],
[0.34912, 0.34167, 0.3355, 0.32967, 0.33701, 0.33288, 0.32827, 0.32432],
[0.34891, 0.34154, 0.33539, 0.3297, 0.33742, 0.33337, 0.32881, 0.32492]]

##### Build Heston Model #####
v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5 # dummy parameters
process = ql.HestonProcess(riskfree_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
model = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model)

##### Build Heston Helpers #####
heston_helpers = []
for i, expiration_date in enumerate(expiration_dates):
    for j, strike in enumerate(strikes):
        tenor = ql.Period(calendar.businessDaysBetween(calculation_date, expiration_date), ql.Days)
        vol_quote = ql.QuoteHandle(ql.SimpleQuote(implied_vols[i][j]))
        helper = ql.HestonModelHelper(tenor, calendar, spot_price, strike, vol_quote, riskfree_ts, dividend_ts)
        helper.setPricingEngine(engine)
        heston_helpers.append(helper)

##### Model Calibration #####
lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
model.calibrate(heston_helpers, lm, ql.EndCriteria(500, 50, 1.0e-8, 1.0e-8, 1.0e-8))
theta, kappa, sigma, rho, v0 = model.params()
print("theta = %f kappa = %f sigma = %f rho = %f v0 = %f" % (theta, kappa, sigma, rho, v0))
print ("-"*70)

def heston_iv (strike, expiration_date):
    ##### Construct European option #####
    option_type = ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.EuropeanExercise(expiration_date)
    european_option = ql.VanillaOption(payoff, exercise)

    ##### Construct and price BSM process #####
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, riskfree_ts, flat_vol_ts)
    engine = ql.AnalyticHestonEngine(model)
    european_option.setPricingEngine(engine)
    bsm_iv = european_option.impliedVolatility(european_option.NPV(), bsm_process)
    
    return bsm_iv

##### Model Validation #####
avg = 0.0
avg_iv = 0.0
print ("%15s %25s %15s %15s %15s %15s %20s %25s" % ("Strikes", "Maturity Date", "Market Value", "Model Value", "Market IV", "Model IV", "Relative Error (%)", "Relative IV Error (%)"))
print ("="*70)
for k, opt in enumerate(heston_helpers):
    err = (opt.modelValue()/opt.marketValue() - 1.0)
    avg += abs(err)
    i = k//len(strikes)
    j = k%len(strikes)
    expiration_date = expiration_dates[i]
    strike = strikes[j]
    h_iv = heston_iv (strike, expiration_date)
    errIV = (h_iv/implied_vols[i][j] - 1.0)
    avg_iv += abs(errIV)
    print ("%15.2f %25s %14.5f %15.5f %15.5f %15.5f %20.7f %25.7f " % (strike, expiration_date, opt.marketValue(), opt.modelValue(), implied_vols[i][j], h_iv, 100.0*err, 100*errIV))
avg = avg*100.0/len(heston_helpers)
avg_iv = avg_iv*100.0/len(heston_helpers)
print ("-"*70)
print ("Average Abs Price Error (%%) : %5.3f" % (avg))
print ("Average Abs IV Error (%%) : %5.3f" % (avg_iv))

# payoff = ql.PlainVanillaPayoff(ql.Option.Put, strikes[3])
# exercise = ql.EuropeanExercise(expiration_date)
# european_option = ql.VanillaOption(payoff, exercise)
# european_option.setPricingEngine(engine)
# print ("Price using VanillaOption = %2.4f" %european_option.NPV())