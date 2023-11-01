############################################################
###
### Generate training data for NN
### Type: params-IVS pairs
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

##### Option Conventions #####
moneyness = [.8, .85, .9, .95, .975, .99, 1.0, 1.01, 1.025, 1.05, 1.1, 1.15, 1.2] # S_0/K 
tenors = ['1W', '2W', '1M', '2M', '3M', '6M', '1Y'] # T

################## Helper Functions ##################

##### Create Implied Vol Data Structure #####
def create_imp_vol_skeleton(moneyness_struct, tenor_struct, calculation_date, spot_price):
    strike_prices = [spot_price/m for m in moneyness_struct]
    expiration_dates = [calculation_date + ql.Period(tenor) for tenor in tenor_struct]
    ttm_days = [(d-calculation_date) for d in expiration_dates] # time to maturity
    ttm_year = [day_count.yearFraction(calculation_date, d) for d in expiration_dates]
    
    new_array = np.array((ttm_days,strike_prices),dtype=object)
    cartesian_product_vola_surface = list(product(*new_array))
    df = pd.DataFrame(cartesian_product_vola_surface, columns=['ttm_days','strike_price'])
    df['moneyness'] = [spot_price / strike_price for strike_price in df['strike_price'] ]
    return strike_prices, np.array((ttm_year)), expiration_dates, df

##### Create European Options #####
def create_eu_options(calculation_date, option_type, strike_price, ttm_days):
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(calculation_date + ttm_days)
    eu_option = ql.VanillaOption(payoff, exercise)
    return eu_option

################## Heston Model Class ##################

class Heston_Model:
    def __init__(self, spot_price = 100, risk_free_rate = 0.03, dividend_rate = 0.01, option_type = ql.Option.Call,
                calculation_date = ql.Date(1, 1, 2019), moneyness = moneyness, tenors = tenors, 
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

        ##### Create Implied Vol Data Structure #####
        self.strike_prices, self.ttm_days, self.expiration_dates, self.df = create_imp_vol_skeleton(moneyness, tenors, calculation_date, spot_price)
        
        ##### Construct Heston Process #####
        process = ql.HestonProcess(self.riskfree_ts, self.dividend_ts, self.spot_handle, self.v0, self.kappa, self.theta, self.sigma, self.rho)
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)

        ##### Create European Options #####
        eu_options = [create_eu_options(calculation_date, option_type, strike_price, ttm_days) 
                        for strike_price, ttm_days in zip(self.df['strike_price'], self.df['ttm_days'])]

        ##### Option Pricing #####
        [opt.setPricingEngine(engine) for opt in eu_options]
        self.df['price'] = [opt.NPV() for opt in eu_options]
        bsm_process = ql.BlackScholesMertonProcess(self.spot_handle, self.dividend_ts, self.riskfree_ts, 
                                                    ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, math.sqrt(v0), day_count))) ##dummy BSM process to calculate implied vol
        self.df['iv'] = [opt.impliedVolatility(opt.NPV(), bsm_process) for opt in eu_options]

    def model_setting(self):
        return [self.risk_free_rate, self.dividend_rate, self.v0, self.kappa, self.theta, self.rho, self.sigma]
    
    def implied_vol_surface(self):
        return self.df['iv'].to_list()

################## Helper Class to Generate Random Heston Paramters ##################

param_bound = {
    'risk_free_rate': [0, 0.05],
    'dividend_rate': [0, 0.02],
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
        params = [0.03, 0.01, 0.04, 0.1, 0.04, -0.5, 0.1]
        for i, key in enumerate(param_bound):
            if key == 'v0' or key == 'theta':
                params[i] = np.random.uniform(low = math.sqrt(self.param_bound[key][0]), high = math.sqrt(self.param_bound[key][1])) ** 2
            else:
                params[i] = np.random.uniform(low = self.param_bound[key][0], high = self.param_bound[key][1])
        return Heston_Model(risk_free_rate = params[0], dividend_rate = params[1],
                            v0 = params[2], kappa = params[3], theta = params[4], rho = params[5], sigma = params[6])
    
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
                x.append(heston_obj.model_setting())
                y.append(heston_obj.implied_vol_surface())
                i += 1

        if ((save_dir is not None) and (type(save_dir) == str)):
            np.save(save_dir+'_x', x)
            np.save(save_dir+'_y', y)
        
        return x, y

Rand_Helper = Rand_Heston_Helper()
x,y = Rand_Helper.train_data_gen(n = 500000, save_dir = './data/test1_20210210')
