############################################################
###
### Generate training data for NN
### Type: params-IV pairs
###
############################################################

##### Import Modules #####
import math
import pandas as pd
import numpy as np
from itertools import product 
import QuantLib as ql

################## Initinal Settings ##################

##### Set Current Time #####
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
calculation_date = ql.Date(8, 2, 2019)


################## Helper Functions ##################

##### Create European Options #####
def create_eu_option(calculation_date, option_type, strike_price, expiration_date):
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(expiration_date)
    eu_option = ql.VanillaOption(payoff, exercise)
    return eu_option

################## Heston Model Class ##################

class Heston_Model:
    def __init__(self, spot_price = 100, risk_free_rate = 0.03, dividend_rate = 0.01, option_type = ql.Option.Call,
                calculation_date = ql.Date(1, 1, 2019), moneyness = 1, ttm_days = 30, 
                v0 = 0.04, kappa = 0.1, theta = 0.04, rho = -0.5, sigma = 0.1):
        
        self.option_type = option_type
        self.calculation_date = calculation_date
        ql.Settings.instance().evaluationDate = self.calculation_date

        ##### Set up Heston Parameters #####
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.sigma = sigma

        ##### Set up Quote Handle and Term Structure #####
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_rate = dividend_rate
        self.spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        self.riskfree_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))

        ##### Set up Option Settings #####
        self.moneyness = moneyness
        self.strike_price = spot_price / moneyness
        self.expiration_date = calculation_date + ttm_days
        self.ttm_days = self.expiration_date - calculation_date
        
        ##### Construct Heston Process #####
        process = ql.HestonProcess(self.riskfree_ts, self.dividend_ts, self.spot_handle, self.v0, self.kappa, self.theta, self.sigma, self.rho)
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)

        ##### Create European Options #####
        opt = create_eu_option(self.calculation_date, self.option_type, self.strike_price, self.expiration_date)

        ##### Option Pricing #####
        opt.setPricingEngine(engine)
        self.price = opt.NPV()
        bsm_process = ql.BlackScholesMertonProcess(self.spot_handle, self.dividend_ts, self.riskfree_ts, 
                                                    ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, math.sqrt(v0), day_count))) ##dummy BSM process to calculate implied vol
        self.iv = opt.impliedVolatility(opt.NPV(), bsm_process)

    def option_setting(self):
        return [self.risk_free_rate, self.dividend_rate, self.moneyness, self.ttm_days, self.v0, self.kappa, self.theta, self.rho, self.sigma]
    
    def implied_vol(self):
        return self.iv

################## Helper Class to Generate Random Heston Paramters ##################

param_bound = {
    'risk_free_rate': [0, 0.05],
    'dividend_rate': [0, 0.02],
    'moneyness': [0.8, 1.2],
    'ttm_days': [1, 366],
    'v0': [0.04, 0.36],
    'kappa': [0, 5],
    'theta': [0.04, 0.36],
    'rho': [-0.9, 0],
    'sigma': [0.1, 0.8]
}

class Rand_Heston_Helper:
    def __init__(self, param_bound = param_bound):
        self.param_bound = param_bound
    
    def create_Heston_Model(self):
        params = [0.03, 0.01, 1, 31, 0.04, 0.1, 0.04, -0.5, 0.1]
        for i, key in enumerate(param_bound):
            if key == 'v0' or key == 'theta':
                params[i] = np.random.uniform(low = math.sqrt(self.param_bound[key][0]), high = math.sqrt(self.param_bound[key][1])) ** 2
            else:
                if key == 'ttm_days':
                    params[i] = np.random.randint(low = self.param_bound[key][0], high = self.param_bound[key][1])
                else:
                    params[i] = np.random.uniform(low = self.param_bound[key][0], high = self.param_bound[key][1])
        return Heston_Model(risk_free_rate = params[0], dividend_rate = params[1], moneyness = params[2], ttm_days = params[3],
                            v0 = params[4], kappa = params[5], theta = params[6], rho = params[7], sigma = params[8])
    
    def train_data_gen(self, seed = 0, n = 50000, save_dir = None):
        np.random.seed(seed)
        x = []
        y = []
        i = 0
        while i < n:
            try:
                heston_obj = self.create_Heston_Model()
            except RuntimeError as e:
                print(i,e)
            else:
                x.append(heston_obj.option_setting())
                y.append(heston_obj.implied_vol())
                i += 1

        if ((save_dir is not None) and (type(save_dir) == str)):
            np.save(save_dir+'_params', x)
            np.save(save_dir+'_iv', y)
        
        return x, y

Rand_Helper = Rand_Heston_Helper()
x,y = Rand_Helper.train_data_gen(n = 5000000, save_dir = './data/params_to_iv_train')
