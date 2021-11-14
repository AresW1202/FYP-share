import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


#Import the stock data
ticker = 'TSM'
targetdata = pd.DataFrame()
targetdata[ticker] = wb.DataReader(ticker, data_source = 'yahoo', start = '2009-1-1', end = '2019-12-28')['Adj Close']
targetdata_MA = pd.DataFrame()
targetdata_MA[ticker] = wb.DataReader(ticker, data_source = 'yahoo', start = '2019-1-1', end = '2020-12-28')['Adj Close']

#Plot stock price 
targetmean=np.mean(targetdata)
tragetstd=np.std(targetdata)
targetdata.plot(figsize=(15,6))
plt.title("Stock Price from 2009-1-1 to 2019-12-28 ")
plt.ylabel("Price")
plt.show()

#Plot for 2020 MA and Bollinger Bands
start_date = '2020-01-01'
end_date = '2020-12-28'

short_rolling = targetdata_MA.rolling(window=20).mean()
long_rolling = targetdata_MA.rolling(window=100).mean()

targetdata_std = targetdata_MA.rolling(window = 20).std()
upper_bb = short_rolling + targetdata_std * 2
lower_bb = short_rolling - targetdata_std * 2


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

#Compute and Plot 2020 MA Stretegry
trading_positions_raw = targetdata_MA - short_rolling
trading_positions = trading_positions_raw.apply(np.sign)
trading_positions_final = trading_positions.shift(1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
ax1.title.set_text('Trading Timing in 2020')
ax1.plot(targetdata_MA.loc[start_date:end_date, :], label='Price')
ax1.plot(short_rolling.loc[start_date:end_date, :], label = '20-days SMA')
ax1.set_ylabel('Stock Price')
ax1.legend(loc='best')
ax2.plot(trading_positions_final.loc[start_date:end_date, :], label='Trading position')
ax2.set_ylabel('Trading position')
plt.show()

#Compute and Plot log return of 2020 MA strategy
number_of_years=1
asset_log_returns = np.log(targetdata_MA).diff()
strategy_asset_log_returns = trading_positions_final * asset_log_returns
cum_strategy_asset_log_returns = strategy_asset_log_returns.cumsum()
last_value_cum_strategy_asset_log_returns=strategy_asset_log_returns.sum()
MA_average_yearly_return = (1 + last_value_cum_strategy_asset_log_returns)**(1/number_of_years) - 1
print(MA_average_yearly_return)

buy_and_hold=pd.DataFrame(1, index = targetdata_MA.index, columns=targetdata_MA.columns)
buy_and_hold_log_returns=buy_and_hold * asset_log_returns
cum_buy_and_hold_log_returns = buy_and_hold_log_returns.cumsum()
last_value_cum_buy_and_hold_log_returnss=buy_and_hold_log_returns.sum()
buy_and_hold_average_yearly_return = (1 + last_value_cum_buy_and_hold_log_returnss)**(1/number_of_years) - 1
print(buy_and_hold_average_yearly_return)

fig = plt.figure(figsize=(16,9))
ax1 = plt.subplot2grid(shape=(16, 9), loc=(0, 0), rowspan=9,colspan=16)
ax1.plot(cum_strategy_asset_log_returns.loc[start_date:end_date, :], label='EMA strategy')
ax1.plot(cum_buy_and_hold_log_returns.loc[start_date:end_date, :], label='Buy and hold')
ax1.title.set_text('Cumulative log-returns using 20MA in 2020')
ax1.set_ylabel('Cumulative log-returns ')
ax1.legend(loc='best')

ax2 = plt.subplot2grid(shape=(8, 5), loc=(6, 0), rowspan=1)
column_labels=["Yearly Return"]
row_labels=["20MA","Buy and Hold"]
ax2.title.set_text('Yearly return of different strategy in 2020 (%)')
data=[[100*np.float(MA_average_yearly_return.iloc[0])],[100*np.float(buy_and_hold_average_yearly_return.iloc[0])]]
ax2.table(cellText=data,colLabels=column_labels,rowLabels=row_labels,loc='center')
ax2.axis('tight')
ax2.axis('off')
plt.tight_layout()
plt.show()

#Compute the stock logarithmic returns (MCS)
log_returns = np.log(1 + targetdata.pct_change())
#Plot  (MCS)
sns.distplot(log_returns.iloc[1:])
plt.title("Frequency of Daily Return from 2009-1-1 to 2019-12-28 ")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

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
plt.figure(figsize=(15,6))
plt.plot(pd.DataFrame(price_paths).iloc[:,0:1000])
plt.title("Stock Price Forecast in 2020 ")
plt.xlabel("Day")
plt.ylabel("Price")
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

