############################################################
###
### Query market IVS data
### Stock: FB
###
############################################################

import numpy as np
import pandas as pd
import datetime
import QuantLib as ql

from yahoo_fin import options

pd.options.mode.chained_assignment = None

spot_price = 257.62
ticker = 'fb'
end_trade_time = datetime.datetime(2021, 2, 26, 16, 0, 0)

moneyness = [.8, .85, .9, .95, .975, .99, 1.0, 1.01, 1.025, 1.05, 1.1, 1.15, 1.2] # S_0/K 

def df_process(df):
    df['Last Trade Date'] = pd.to_datetime(df['Last Trade Date'])
    df = df[(end_trade_time - df['Last Trade Date']).dt.total_seconds() <= 3600]
    df[['Volume', 'Strike', 'Bid', 'Ask']] = df[['Volume', 'Strike', 'Bid', 'Ask']].apply(pd.to_numeric)
    df['Implied Volatility'] = df['Implied Volatility'].str.rstrip('%').astype('float') / 100.0
    return df

def spread(df):
    return (df['Ask'] - df['Bid']) / ((df['Ask'] + df['Bid']) / 2) 

def intra_iv(df_call, df_put, moneyness):
    ivc = []
    for m in moneyness:
        strike_price = spot_price / m
        if m < 1:
            df = df_call
        else:
            df = df_put
        df1 = df[(df['Strike'] <= strike_price) & (abs(df['Strike'] - strike_price) / strike_price < 0.03) & (df['Last Price'] > 0.5) & (df['Last Price'] >= df['Bid']) & (df['Last Price'] <= df['Ask'])]
        df2 = df[(df['Strike'] >= strike_price) & (abs(df['Strike'] - strike_price) / strike_price < 0.03) & (df['Last Price'] > 0.5) & (df['Last Price'] >= df['Bid']) & (df['Last Price'] <= df['Ask'])]
        if (df1.shape[0] == 0) & (df2.shape[0] == 0):
            ivc.append(0)
            continue
        if df1.shape[0] == 0:
            iv2 = df2.iloc[0]
            if abs(iv2['Strike'] - strike_price) / strike_price < 0.005:
                ivc.append(iv2['Implied Volatility'])
            else:
                ivc.append(0)
            continue
        if df2.shape[0] == 0:
            iv1 = df1.iloc[-1]
            if abs(iv1['Strike'] - strike_price) / strike_price < 0.005:
                ivc.append(iv1['Implied Volatility'])
            else:
                ivc.append(0)
            continue
        iv1 = df1.iloc[-1]
        iv2 = df2.iloc[0]
        iv = iv1['Implied Volatility'] + (strike_price - iv1['Strike']) / (iv2['Strike'] - iv1['Strike']) * (iv2['Implied Volatility'] - iv1['Implied Volatility'])
        ivc.append(iv)
    return np.array(ivc)

ivs = []

calculation_date = ql.Date(26, 2, 2021)
dates = [['2021-03-05', '2021-03-05'], ['2021-03-12', '2021-03-12'], ['2021-03-26', '2021-03-26'], ['2021-04-16', '2021-04-16'], ['2021-05-21','2021-05-21'], ['2021-07-16', '2021-09-17'], ['2022-01-21', '2022-01-21']]
tenors = ['1W', '2W', '1M', '2M', '3M', '6M', '1Y']
for i in range(len(dates)):
    d = dates[i]
    d0 = ql.Date(d[0], '%Y-%m-%d')
    d1 = ql.Date(d[1], '%Y-%m-%d')
    tenor = tenors[i]
    d_exp = calculation_date + ql.Period(tenor)

    df_call0 = options.get_calls(ticker, d[0])
    np.save('./raw_' + d[0] + '_20210226_call', df_call0)
    df_call0 = df_process(df_call0)
    df_put0 = options.get_puts(ticker, d[0])
    np.save('./raw_' + d[0] + '_20210226_put', df_put0)
    df_put0 = df_process(df_put0)
    ivc0 = intra_iv(df_call0, df_put0, moneyness)

    if d1 - d0 == 0:
        ivs.append(ivc0.tolist())
        continue
    else:
        f0 = (d1 - d_exp) / (d1 - d0)
        f1 = 1 - f0

    df_call1 = options.get_calls(ticker, d[1])
    np.save('./raw_' + d[1] + '_20210226_call', df_call1)
    df_call1 = df_process(df_call1)
    df_put1 = options.get_puts(ticker, d[1])
    np.save('./raw_' + d[1] + '_20210226_put', df_put1)
    df_put1 = df_process(df_put1)
    ivc1 = intra_iv(df_call1, df_put1, moneyness)
    ivc = (ivc0 * f0 + ivc1 * f1) * (ivc0 > 0) * (ivc1 > 0)
    ivs.append(ivc.tolist())


print(ivs)

np.save('./20210226_ivs', ivs)