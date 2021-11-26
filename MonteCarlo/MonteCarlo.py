import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def total_trading_cost_2020(data):
    trading_freq=0
    short_time=0
    for i in range(1,251):
        if(data.iloc[i][0]==-1):
            short_time+=1
        if (data.iloc[i][0]!=data.iloc[i-1][0]):
            trading_freq+=1

    total_trading_cost=(normal_trade_cost*trading_freq)+(short_cost*short_time)
    print("total_trading_cost:",total_trading_cost)
    print("trading_freq:",trading_freq)
    print("short_time:",short_time)
    print("(normal_trade_cost*trading_freq):",(normal_trade_cost*trading_freq))
    print("(short_cost*short_time):",(short_cost*short_time))




#Import the stock data
ticker = 'TSM'
targetdata = pd.DataFrame()
targetdata[ticker] = wb.DataReader(ticker, data_source = 'yahoo', start = '2009-1-1', end = '2019-12-28')['Adj Close']
targetdata_MA = pd.DataFrame()
targetdata_MA[ticker] = wb.DataReader(ticker, data_source = 'yahoo', start = '2020-1-1', end = '2020-12-28')['Adj Close']

#Plot stock price 
targetmean=np.mean(targetdata)
tragetstd=np.std(targetdata)
"""
targetdata.plot(figsize=(15,6))
plt.title("Stock Price from 2009-1-1 to 2019-12-28 ")
plt.ylabel("Price")
plt.show()
"""

#Trading Cost
invest_amount=1000000
short_rate=0.0025
trade_rate=0.01
normal_trade_cost=invest_amount*trade_rate
short_cost=invest_amount*(short_rate/365)

#Plot for 2020 MA and Bollinger Bands
start_date = '2020-01-01'
end_date = '2020-12-28'

short_rolling = targetdata_MA.rolling(window=20).mean()
long_rolling = targetdata_MA.rolling(window=100).mean()

targetdata_MA_std = targetdata_MA.rolling(window = 20).std()
upper_bb = short_rolling + targetdata_MA_std * 2
lower_bb = short_rolling - targetdata_MA_std * 2

"""
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(targetdata_MA.loc[start_date:end_date, :], label='Price')
ax.plot(long_rolling.loc[start_date:end_date, :], label = '100-days SMA')
ax.plot(short_rolling.loc[start_date:end_date, :], label = '20-days SMA')
ax.plot(upper_bb.loc[start_date:end_date, :], label = 'Upper Bollinger Bands',linestyle='dashed')
ax.plot(lower_bb.loc[start_date:end_date, :], label = 'Lower Bollinger Bands',linestyle='dashed')
ax.legend(loc='best')
ax.title.set_text('MA and Bollinger Bands in 2020')
ax.set_ylabel('Price in $')
plt.show()
"""

#Compute and Plot 2020 MA Stretegry
short_time=0
trading_positions_raw = targetdata_MA - short_rolling
trading_positions = trading_positions_raw.apply(np.sign)
trading_positions_final = trading_positions.shift(1)
for i in range(20):
    trading_positions_final.iloc[i][0]=0

print("TTcost",total_trading_cost_2020(trading_positions_final))




"""
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
ax1.title.set_text('MA Trading Timing in 2020')
ax1.plot(targetdata_MA.loc[start_date:end_date, :], label='Price')
ax1.plot(short_rolling.loc[start_date:end_date, :], label = '20-days SMA')
ax1.set_ylabel('Stock Price')
ax1.legend(loc='best')
ax2.plot(trading_positions_final.loc[start_date:end_date, :], label='Trading position')
ax2.set_ylabel('Trading position')
plt.show()
"""

#Compute Bollinger Band Stretegry 2020
BB_trading_pos=[]

for i in range(251):
    if (targetdata_MA.iloc[i][0] > upper_bb.iloc[i][0]):
        BB_trading_pos.append(-1)
    elif (targetdata_MA.iloc[i][0] < lower_bb.iloc[i][0]):
        BB_trading_pos.append(1)
    else:
        BB_trading_pos.append(0)

targetdata_BB=targetdata_MA.copy(deep=True)
targetdata_BB[ticker]=BB_trading_pos
BB_trading_pos_final=targetdata_BB.shift(1)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
ax1.title.set_text('BB Trading Timing in 2020')
ax1.plot(targetdata_MA.loc[start_date:end_date, :], label='Price')
ax1.plot(short_rolling.loc[start_date:end_date, :], label = '20-days SMA')
ax1.plot(upper_bb.loc[start_date:end_date, :], label = 'Upper BB')
ax1.plot(lower_bb.loc[start_date:end_date, :], label = 'Lower BB')
ax1.set_ylabel('Stock Price')
ax1.legend(loc='best')
ax2.plot(BB_trading_pos_final.loc[start_date:end_date, :], label='Trading position')
ax2.set_ylabel('Trading position')
plt.show()

#Compute and Plot log return of 2020 MA/BB/BNH strategy
risk_free_rate=0.001
number_of_years=1
asset_log_returns = np.log(targetdata_MA).diff()
strategy_asset_log_returns = trading_positions_final * asset_log_returns
cum_strategy_asset_log_returns = strategy_asset_log_returns.cumsum()
last_value_cum_strategy_asset_log_returns=strategy_asset_log_returns.sum()
MA_average_yearly_return = (1 + last_value_cum_strategy_asset_log_returns)**(1/number_of_years) - 1
sharpe_ratio_MA= (MA_average_yearly_return-risk_free_rate)/cum_strategy_asset_log_returns.std(0)

buy_and_hold=pd.DataFrame(1, index = targetdata_MA.index, columns=targetdata_MA.columns)
buy_and_hold_log_returns=buy_and_hold * asset_log_returns
cum_buy_and_hold_log_returns = buy_and_hold_log_returns.cumsum()
last_value_cum_buy_and_hold_log_returnss=buy_and_hold_log_returns.sum()
buy_and_hold_average_yearly_return = (1 + last_value_cum_buy_and_hold_log_returnss)**(1/number_of_years) - 1
sharpe_ratio_buy_and_hold= (buy_and_hold_average_yearly_return-risk_free_rate)/cum_buy_and_hold_log_returns.std(0)

BB_log_returns=BB_trading_pos_final* asset_log_returns
cum_BB_log_returns = BB_log_returns.cumsum()
last_value_cum_BB_log_returnss=BB_log_returns.sum()
BB_average_yearly_return = (1 + last_value_cum_BB_log_returnss)**(1/number_of_years) - 1
sharpe_ratio_BB= (BB_average_yearly_return-risk_free_rate)/cum_BB_log_returns.std(0)


fig = plt.figure(figsize=(16,9))
ax1 = plt.subplot2grid(shape=(16, 9), loc=(0, 0), rowspan=9,colspan=16)
ax1.plot(cum_strategy_asset_log_returns.loc[start_date:end_date, :], label='MA strategy')
ax1.plot(cum_buy_and_hold_log_returns.loc[start_date:end_date, :], label='Buy and hold')
ax1.plot(cum_BB_log_returns.loc[start_date:end_date, :], label='BB strategy')
ax1.title.set_text('Cumulative log-returns using 20MA/BB/BNH in 2020')
ax1.set_ylabel('Cumulative log-returns ')
ax1.legend(loc='best')

ax2 = plt.subplot2grid(shape=(8, 5), loc=(6, 0), rowspan=1)
column_labels=["Yearly Return"]
row_labels=["20MA","BB","Buy and Hold"]
ax2.title.set_text('Yearly return of different strategy in 2020 (%)')
data=[[100*np.float(MA_average_yearly_return.iloc[0])],[100*np.float(BB_average_yearly_return.iloc[0])],[100*np.float(buy_and_hold_average_yearly_return.iloc[0])]]
ax2.table(cellText=data,colLabels=column_labels,rowLabels=row_labels,loc='center')
ax2.axis('tight')
ax2.axis('off')

ax3 = plt.subplot2grid(shape=(8, 5), loc=(6, 2), rowspan=1)
column_labels=["Sharpe Ratio"]
row_labels=["20MA","BB","Buy and Hold"]
ax3.title.set_text('Sharpe Ratio of different strategy in 2020')
data=[[sharpe_ratio_MA.iloc[0]],[sharpe_ratio_BB.iloc[0]],[sharpe_ratio_buy_and_hold.iloc[0]]]
ax3.table(cellText=data,colLabels=column_labels,rowLabels=row_labels,loc='center')
ax3.axis('tight')
ax3.axis('off')
plt.tight_layout()
plt.show()


#Compute the stock logarithmic returns (MCS)
log_returns = np.log(1 + targetdata.pct_change())
#Plot  (MCS)
"""
sns.distplot(log_returns.iloc[1:])
plt.title("Frequency of Daily Return from 2009-1-1 to 2019-12-28 ")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()
"""

#Compute the Drift (MCS)
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var)

#Compute the Variance and Daily Returns (MCS)
stdev = log_returns.std()
days = 252
trials = 1000
Z = norm.ppf(np.random.rand(days, trials)) #days, trials
daily_returns = np.exp(drift.values + stdev.values * Z)

#Calculating the stock price for every trial (MCS)
price_paths = np.zeros_like(daily_returns)
price_paths[0] = targetdata.iloc[-1]
for t in range(1, days):
    price_paths[t] = price_paths[t-1]*daily_returns[t]

"""
plt.figure(figsize=(15,6))
plt.plot(pd.DataFrame(price_paths).iloc[:,0:1000])
plt.title("Stock Price Forecast in 2020 ")
plt.xlabel("Day")
plt.ylabel("Price")
plt.show()
"""

#Plot for 2020 MA and Bollinger Bands (MCS)
for i in range(10000):
    targetdata_price_storage_MCS =[]
    for j in range(252):
        targetdata_price_storage_MCS.append(price_paths[j][i])
    targetdata_MA_MCS=pd.DataFrame(targetdata_price_storage_MCS)
    short_rolling_MCS = targetdata_MA_MCS.rolling(window=20).mean()
    long_rolling_MCS = targetdata_MA_MCS.rolling(window=100).mean()

    targetdata_std_MCS = targetdata_MA_MCS.rolling(window = 20).std()
    upper_bb_MCS = short_rolling_MCS + targetdata_std_MCS * 2
    lower_bb_MCS = short_rolling_MCS - targetdata_std_MCS * 2

    
    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(targetdata_MA_MCS.iloc[:252], label='Price')
    ax.plot(long_rolling_MCS.iloc[:252], label = '100-days SMA')
    ax.plot(short_rolling_MCS.iloc[:252], label = '20-days SMA')
    ax.plot(upper_bb_MCS.iloc[:252], label = 'Upper Bollinger Bands',linestyle='dashed')
    ax.plot(lower_bb_MCS.iloc[:252], label = 'Lower Bollinger Bands',linestyle='dashed')
    ax.legend(loc='best')
    ax.title.set_text('MA and Bollinger Bands in 2020 using MCS')
    ax.set_ylabel('Price in $')
    plt.show()
    

    #Compute and Plot 2020 MA Stretegry (MCS)
    trading_frequency=0
    trading_positions_raw_MCS = targetdata_MA_MCS - short_rolling_MCS 
    trading_positions_MCS = trading_positions_raw_MCS.apply(np.sign)
    trading_positions_final_MCS = trading_positions_MCS.shift(1)
    for i in range (20):
        trading_positions_final_MCS[0][i]=0
    for i in range (21,252):
        if (trading_positions_final_MCS[0][i]!=trading_positions_final_MCS[0][i-1]):
            trading_frequency +=1
    print("MCS trad MA freq",trading_frequency)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
    ax1.title.set_text('Trading Timing in 2020 using MCS')
    ax1.plot(targetdata_MA_MCS.iloc[:252], label='Price')
    ax1.plot(short_rolling_MCS.loc[:252], label = '20-days SMA')
    ax1.set_ylabel('Stock Price')
    ax1.legend(loc='best')
    ax2.plot(trading_positions_final_MCS.iloc[:252], label='Trading position')
    ax2.set_ylabel('Trading position')
    plt.show()

  
    #Compute Bollinger Band Stretegry2020 (MCS)
    BB_trading_pos_MCS=[]
    BB_trading_pos_final_MCS=[]
    for k in range(252):
        if (targetdata_price_storage_MCS[k] > upper_bb_MCS.iloc[k][0]):
            BB_trading_pos_MCS.append(-1)
        elif (targetdata_price_storage_MCS[k] < lower_bb_MCS.iloc[k][0]):
            BB_trading_pos_MCS.append(1)
        else:
            BB_trading_pos_MCS.append(0)

    BB_trading_pos_MCS.roll(1)
    BB_trading_pos_MCS[0]=BB_trading_pos_MCS[0]*0


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
    ax1.title.set_text('BB Trading Timing in 2020 using MCS')
    ax1.plot(targetdata_price_storage_MCS[0:252], label='Price')
    ax1.plot(short_rolling_MCS.iloc[:252], label = '20-days SMA')
    ax1.plot(upper_bb_MCS.iloc[:252], label = 'Upper BB')
    ax1.plot(lower_bb_MCS.iloc[:252], label = 'Lower BB')
    ax1.set_ylabel('Stock Price')
    ax1.legend(loc='best')
    ax2.plot(BB_trading_pos_MCS[0:252], label='Trading position')
    ax2.set_ylabel('Trading position')
    plt.show()
    
    #Compute and Plot log return of 2020 MA strategy (MCS)
    number_of_years=1
    asset_log_returns_MCS = np.log(targetdata_MA_MCS).diff()
    strategy_asset_log_returns_MCS = trading_positions_final_MCS * asset_log_returns_MCS
    cum_strategy_asset_log_returns_MCS = strategy_asset_log_returns_MCS.cumsum()
    last_value_cum_strategy_asset_log_returns_MCS=strategy_asset_log_returns_MCS.sum()
    MA_average_yearly_return_MCS = (1 + last_value_cum_strategy_asset_log_returns_MCS)**(1/number_of_years) - 1

    buy_and_hold_MCS=pd.DataFrame(1, index = targetdata_MA_MCS.index, columns=targetdata_MA_MCS.columns)
    buy_and_hold_log_returns_MCS=buy_and_hold_MCS * asset_log_returns_MCS
    cum_buy_and_hold_log_returns_MCS = buy_and_hold_log_returns_MCS.cumsum()
    last_value_cum_buy_and_hold_log_returnss_MCS=buy_and_hold_log_returns_MCS.sum()
    buy_and_hold_average_yearly_return_MCS = (1 + last_value_cum_buy_and_hold_log_returnss_MCS)**(1/number_of_years) - 1

    fig = plt.figure(figsize=(16,9))
    ax1 = plt.subplot2grid(shape=(16, 9), loc=(0, 0), rowspan=9,colspan=16)
    ax1.plot(cum_strategy_asset_log_returns_MCS.iloc[:252], label='MA strategy')
    ax1.plot(cum_buy_and_hold_log_returns_MCS.iloc[:252], label='Buy and hold')
    ax1.title.set_text('Cumulative log-returns using 20MA in 2020 using MCS')
    ax1.set_ylabel('Cumulative log-returns ')
    ax1.legend(loc='best')

    ax2 = plt.subplot2grid(shape=(8, 5), loc=(6, 0), rowspan=1)
    column_labels=["Yearly Return"]
    row_labels=["20MA","Buy and Hold"]
    ax2.title.set_text('Yearly return of different strategy in 2020 (%)')
    data=[[100*np.float(MA_average_yearly_return_MCS.iloc[0])],[100*np.float(buy_and_hold_average_yearly_return_MCS.iloc[0])]]
    ax2.table(cellText=data,colLabels=column_labels,rowLabels=row_labels,loc='center')
    ax2.axis('tight')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

#Calculating the paths stat (MCS)
pathsmean=np.mean(price_paths)
pathsstd=np.std(price_paths)
data=[[np.float(pathsmean),np.float(pathsstd)]]
column_labels=["mean", "std"]
plt.axis('tight')
plt.axis('off')
plt.table(cellText=data,colLabels=column_labels,loc='center')
plt.show()

#period static 
period_stat=[]
for p in range(1,days):
    period_stat=price_paths[p]
period_mean=np.mean(period_stat)
period_std=np.std(period_stat)
data=[[np.float(period_mean),np.float(period_std)]]
column_labels=["mean", "std"]
plt.title('stat from Feb to Mar')
plt.axis('tight')
plt.axis('off')
plt.table(cellText=data,colLabels=column_labels,loc='center')
plt.show()


#Histogram (MCS)
sns.distplot(pd.DataFrame(price_paths).iloc[-1])
plt.title("Stock Price Forecast in 2020 ")
plt.ylabel("Probability")
plt.xlabel("Price in 252 days")
plt.show()
