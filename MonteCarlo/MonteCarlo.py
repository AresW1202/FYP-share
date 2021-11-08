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
#Plot
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

#Compute the logarithmic returns
log_returns = np.log(1 + targetdata.pct_change())
#Plot
sns.distplot(log_returns.iloc[1:])
plt.title("Frequency of Daily Return from 2009-1-1 to 2019-12-28 ")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

#Compute the Drift
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var)

#Compute the Variance and Daily Returns
stdev = log_returns.std()
days = 252
trials = 1000
Z = norm.ppf(np.random.rand(days, trials)) #days, trials
daily_returns = np.exp(drift.values + stdev.values * Z)

#Calculating the stock price for every trial
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

#Calculating the paths stat
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


#Histogram 
sns.distplot(pd.DataFrame(price_paths).iloc[-1])
plt.title("Stock Price Forecast in 2020 ")
plt.ylabel("Probability")
plt.xlabel("Price in 252 days")
plt.show()


