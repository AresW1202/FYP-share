import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def log_return_after_cost_2020(data,pos):
    Pre_Cost=[]
    Trade_Cost=[]
    Post_Cost=[]
    Return=[]
    Current_Value=[]
    for i in range(251):
        Pre_Cost.append(0)
        Trade_Cost.append(0)
        Post_Cost.append(0)
        Return.append(0)
        Current_Value.append(1000000)
    for i in range(20):
        pos[i]=0

    for i in range(20,251):
        if(pos[i]!=pos[i-1] and pos[i]==-1):
            Pre_Cost[i]=Current_Value[i-1]
            Trade_Cost[i]=Pre_Cost[i]*0.0025/365+Pre_Cost[i]*0.01
            Post_Cost[i]=Pre_Cost[i]-Trade_Cost[i]
            Return[i]=pos[i]*np.log(data.iloc[i][0]/data.iloc[i-1][0])
            Current_Value[i]=Post_Cost[i]*(1+Return[i])
        elif (pos[i]!=pos[i-1]):
            Pre_Cost[i]=Current_Value[i-1]
            Trade_Cost[i]=Pre_Cost[i]*0.01
            Post_Cost[i]=Pre_Cost[i]-Trade_Cost[i]
            Return[i]=pos[i]*np.log(data.iloc[i][0]/data.iloc[i-1][0])
            Current_Value[i]=Post_Cost[i]*(1+Return[i])
        elif (pos[i]==-1):
            Pre_Cost[i]=Current_Value[i-1]
            Trade_Cost[i]=Pre_Cost[i]*0.0025/365
            Post_Cost[i]=Pre_Cost[i]-Trade_Cost[i]
            Return[i]=pos[i]*np.log(data.iloc[i][0]/data.iloc[i-1][0])
            Current_Value[i]=Post_Cost[i]*(1+Return[i])
        else:
             Pre_Cost[i]=np.nan
             Trade_Cost[i]=np.nan
             Post_Cost[i]=Current_Value[i-1]
             Return[i]=pos[i]*np.log(data.iloc[i][0]/data.iloc[i-1][0])
             Current_Value[i]=Post_Cost[i]*(1+Return[i])

    data['Pos']=pos
    data['Pre_Cost']=Pre_Cost
    data['Trade_Cost']=Trade_Cost
    data['Post_Cost']=Post_Cost
    data['Return']=Return
    data['Current_Value']=Current_Value

def Sharpe_ratio(year_return,data):
    risk_free_rate=0.001
    Sharpe_ratio= (year_return-risk_free_rate)/np.std(data)
    return Sharpe_ratio


#Import the stock data
ticker = 'TSM'
targetdata = pd.DataFrame()
targetdata[ticker] = wb.DataReader(ticker, data_source = 'yahoo', start = '2009-1-1', end = '2019-12-28')['Adj Close']
Targetdata_2020 = pd.DataFrame()
Targetdata_2020[ticker] = wb.DataReader(ticker, data_source = 'yahoo', start = '2020-1-1', end = '2020-12-28')['Adj Close']

#Plot stock price 
targetmean=np.mean(targetdata)
tragetstd=np.std(targetdata)
"""
targetdata.plot(figsize=(15,6))
plt.title("Stock Price from 2009-1-1 to 2019-12-28 ")
plt.ylabel("Price")
plt.show()
"""


#Plot for 2020 MA and Bollinger Bands
start_date = '2020-01-01'
end_date = '2020-12-28'

short_rolling = Targetdata_2020.rolling(window=20).mean()
long_rolling = Targetdata_2020.rolling(window=100).mean()

Targetdata_2020_std = Targetdata_2020.rolling(window = 20).std()
upper_bb = short_rolling + Targetdata_2020_std * 2
lower_bb = short_rolling - Targetdata_2020_std * 2

"""
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(Targetdata_2020.loc[start_date:end_date, :], label='Price')
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
Targetdata_2020_MA=Targetdata_2020.copy(deep=True)
trading_positions_raw = Targetdata_2020 - short_rolling
trading_positions = trading_positions_raw.apply(np.sign)
trading_positions_final = trading_positions.shift(1)
trading_positions_final_idx=[]
for i in range(20):
    trading_positions_final.iloc[i,0]=0
for i in range(251):
    trading_positions_final_idx.append(trading_positions_final.iloc[i][0])
log_return_after_cost_2020(Targetdata_2020_MA,trading_positions_final_idx)

"""
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
ax1.title.set_text('MA Trading Timing in 2020')
ax1.plot(Targetdata_2020.loc[start_date:end_date, :], label='Price')
ax1.plot(short_rolling.loc[start_date:end_date, :], label = '20-days SMA')
ax1.set_ylabel('Stock Price')
ax1.legend(loc='best')
ax2.plot(trading_positions_final.loc[start_date:end_date, :], label='Trading position')
ax2.set_ylabel('Trading position')
plt.show()
"""

#Compute Bollinger Band Stretegry 2020
BB_trading_pos=[]
for i in range(20):
    BB_trading_pos.append(0)
for i in range(20,251):
    if (Targetdata_2020.iloc[i][0] > upper_bb.iloc[i][0]):
        BB_trading_pos.append(-1)
    elif (Targetdata_2020.iloc[i][0] < lower_bb.iloc[i][0]):
        BB_trading_pos.append(1)
    else:
        BB_trading_pos.append(BB_trading_pos[i-1])


Targetdata_2020_BB=Targetdata_2020.copy(deep=True)
targetdata_2020_BB_dump_df=Targetdata_2020.copy(deep=True)
targetdata_2020_BB_dump_df[ticker]=BB_trading_pos
BB_trading_pos_final=targetdata_2020_BB_dump_df.shift(1)
BB_trading_pos_final_idx=[]
for i in range(251):
    BB_trading_pos_final_idx.append(BB_trading_pos_final.iloc[i][0])
log_return_after_cost_2020(Targetdata_2020_BB,BB_trading_pos_final_idx)

"""
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
ax1.title.set_text('BB Trading Timing in 2020')
ax1.plot(Targetdata_2020.loc[start_date:end_date, :], label='Price')
ax1.plot(short_rolling.loc[start_date:end_date, :], label = '20-days SMA')
ax1.plot(upper_bb.loc[start_date:end_date, :], label = 'Upper BB')
ax1.plot(lower_bb.loc[start_date:end_date, :], label = 'Lower BB')
ax1.set_ylabel('Stock Price')
ax1.legend(loc='best')
ax2.plot(BB_trading_pos_final.loc[start_date:end_date, :], label='Trading position')
ax2.set_ylabel('Trading position')
plt.show()
"""

#Compute and Plot log return pre cost of 2020 MA/BB/BNH strategy
risk_free_rate=0.001
number_of_years=1
asset_log_returns = np.log(Targetdata_2020).diff()
strategy_asset_log_returns = trading_positions_final * asset_log_returns
cum_strategy_asset_log_returns = strategy_asset_log_returns.cumsum()
last_value_cum_strategy_asset_log_returns=strategy_asset_log_returns.sum()
MA_average_yearly_return = (1 + last_value_cum_strategy_asset_log_returns)**(1/number_of_years) - 1
sharpe_ratio_MA= (MA_average_yearly_return-risk_free_rate)/cum_strategy_asset_log_returns.std(0)

buy_and_hold=pd.DataFrame(1, index = Targetdata_2020.index, columns=Targetdata_2020.columns)
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
ax1.title.set_text('Cumulative log-returns using 20MA/BB/BNH in 2020 (Without Trading Cost)')
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

#Compute and Plot log return post cost of 2020 MA/BB/BNH strategy
Targetdata_2020_BNH=cum_buy_and_hold_log_returns.copy(deep=True)
Targetdata_2020_BNH=(1+Targetdata_2020_BNH)*1000000
Targetdata_2020_BNH.iloc[1][0]=1000000

Targetdata_2020_MA_year_return=(Targetdata_2020_MA.iloc[-1][6]/Targetdata_2020_MA.iloc[0][6])-1
Targetdata_2020_BB_year_return=(Targetdata_2020_BB.iloc[-1][6]/Targetdata_2020_BB.iloc[0][6])-1
Targetdata_2020_BNH_year_return=(Targetdata_2020_BNH.iloc[-1][0]/Targetdata_2020_BNH.iloc[1][0])-1

fig = plt.figure(figsize=(16,9))
ax1 = plt.subplot2grid(shape=(16, 9), loc=(0, 0), rowspan=9,colspan=16)
ax1.plot(Targetdata_2020_MA.loc[start_date:end_date,'Current_Value' :], label='MA strategy')
ax1.plot(Targetdata_2020_BNH.loc[start_date:end_date,:], label='Buy and hold')
ax1.plot(Targetdata_2020_BB.loc[start_date:end_date,'Current_Value' :], label='BB strategy')
ax1.title.set_text('Assets Value using 20MA/BB/BNH in 2020 (With Trading Cost)')
ax1.set_ylabel('Assets Value')
ax1.legend(loc='best')

ax2 = plt.subplot2grid(shape=(8, 5), loc=(6, 0), rowspan=1)
column_labels=["Yearly Return"]
row_labels=["20MA","BB","Buy and Hold"]
ax2.title.set_text('Yearly return of different strategy in 2020 With Trading Cost (%)')
data=[[100*np.float(Targetdata_2020_MA_year_return)],
      [100*np.float(Targetdata_2020_BB_year_return)],
      [100*np.float(Targetdata_2020_BNH_year_return)]]
ax2.table(cellText=data,colLabels=column_labels,rowLabels=row_labels,loc='center')
ax2.axis('tight')
ax2.axis('off')

ax3 = plt.subplot2grid(shape=(8, 5), loc=(6, 2), rowspan=1)
column_labels=["Sharpe Ratio"]
row_labels=["20MA","BB","Buy and Hold"]
ax3.title.set_text('Sharpe Ratio of different strategy in 2020')
data=[[Sharpe_ratio(Targetdata_2020_MA_year_return,Targetdata_2020_MA['Current_Value'])],
      [Sharpe_ratio(Targetdata_2020_BB_year_return,Targetdata_2020_BB['Current_Value'])],
      [Sharpe_ratio(Targetdata_2020_BNH_year_return,Targetdata_2020_BNH.iloc[:,0])]]
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
days = 251
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
Targetdata_2020_MCS=Targetdata_2020.copy(deep=True)
Each_trial_post_cost_MA_return=[]
Each_trial_post_cost_BB_return=[]
Each_trial_post_cost_BNH_return=[]
for i in range(10000):
    print("sim_time: ",i)
    targetdata_price_storage_MCS =[]
    for j in range(251):
        targetdata_price_storage_MCS.append(price_paths[j][i])

    Targetdata_2020_MCS[ticker]=targetdata_price_storage_MCS
    short_rolling_MCS = Targetdata_2020_MCS.rolling(window=20).mean()
    long_rolling_MCS = Targetdata_2020_MCS.rolling(window=100).mean()
    targetdata_std_MCS = Targetdata_2020_MCS.rolling(window = 20).std()
    upper_bb_MCS = short_rolling_MCS + targetdata_std_MCS * 2
    lower_bb_MCS = short_rolling_MCS - targetdata_std_MCS * 2

    """
    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(Targetdata_2020_MCS.iloc[:251], label='Price')
    ax.plot(long_rolling_MCS.iloc[:251], label = '100-days SMA')
    ax.plot(short_rolling_MCS.iloc[:251], label = '20-days SMA')
    ax.plot(upper_bb_MCS.iloc[:251], label = 'Upper Bollinger Bands',linestyle='dashed')
    ax.plot(lower_bb_MCS.iloc[:251], label = 'Lower Bollinger Bands',linestyle='dashed')
    ax.legend(loc='best')
    ax.title.set_text('MA and Bollinger Bands in 2020 using MCS')
    ax.set_ylabel('Price in $')
    plt.show()
    """

    #Compute and Plot 2020 MA Stretegry (MCS)
    Targetdata_2020_MA_MCS=Targetdata_2020_MCS.copy(deep=True)
    trading_positions_raw = Targetdata_2020_MCS - short_rolling_MCS
    trading_positions = trading_positions_raw.apply(np.sign)
    trading_positions_final = trading_positions.shift(1)
    trading_positions_final_idx=[]
    for i in range(20):
        trading_positions_final.iloc[i,0]=0
    for i in range(251):
        trading_positions_final_idx.append(trading_positions_final.iloc[i][0])
    log_return_after_cost_2020(Targetdata_2020_MA_MCS,trading_positions_final_idx)

    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
    ax1.title.set_text('MA Trading Timing using MCS in 2020')
    ax1.plot(Targetdata_2020_MCS.loc[start_date:end_date, :], label='Price')
    ax1.plot(short_rolling_MCS.loc[start_date:end_date, :], label = '20-days SMA')
    ax1.set_ylabel('Stock Price')
    ax1.legend(loc='best')
    ax2.plot(trading_positions_final.loc[start_date:end_date, :], label='Trading position')
    ax2.set_ylabel('Trading position')
    plt.show()
    """

  
    #Compute Bollinger Band Stretegry2020 (MCS)
    BB_trading_pos=[]
    for i in range(20):
        BB_trading_pos.append(0)
    for i in range(20,251):
        if (Targetdata_2020_MCS.iloc[i][0] > upper_bb.iloc[i][0]):
            BB_trading_pos.append(-1)
        elif (Targetdata_2020_MCS.iloc[i][0] < lower_bb.iloc[i][0]):
            BB_trading_pos.append(1)
        else:
            BB_trading_pos.append(BB_trading_pos[i-1])


    Targetdata_2020_BB_MCS=Targetdata_2020_MCS.copy(deep=True)
    targetdata_2020_BB_dump_df_MCS=Targetdata_2020_MCS.copy(deep=True)
    targetdata_2020_BB_dump_df_MCS[ticker]=BB_trading_pos
    BB_trading_pos_final_MCS=targetdata_2020_BB_dump_df_MCS.shift(1)
    BB_trading_pos_final_idx=[]
    for i in range(251):
        BB_trading_pos_final_idx.append(BB_trading_pos_final_MCS.iloc[i][0])
    log_return_after_cost_2020(Targetdata_2020_BB_MCS,BB_trading_pos_final_idx)

    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
    ax1.title.set_text('BB Trading Timing using MCS in 2020')
    ax1.plot(Targetdata_2020_MCS.loc[start_date:end_date, :], label='Price')
    ax1.plot(short_rolling_MCS.loc[start_date:end_date, :], label = '20-days SMA')
    ax1.plot(upper_bb_MCS.loc[start_date:end_date, :], label = 'Upper BB')
    ax1.plot(lower_bb_MCS.loc[start_date:end_date, :], label = 'Lower BB')
    ax1.set_ylabel('Stock Price')
    ax1.legend(loc='best')
    ax2.plot(BB_trading_pos_final_MCS.loc[start_date:end_date, :], label='Trading position')
    ax2.set_ylabel('Trading position')
    plt.show()
    """

    #Compute and Plot log return pre cost of 2020 MA/BB/BNH strategy
    risk_free_rate=0.001
    number_of_years=1
    asset_log_returns = np.log(Targetdata_2020_MCS).diff()
    strategy_asset_log_returns = trading_positions_final * asset_log_returns
    cum_strategy_asset_log_returns = strategy_asset_log_returns.cumsum()
    last_value_cum_strategy_asset_log_returns=strategy_asset_log_returns.sum()
    MA_average_yearly_return = (1 + last_value_cum_strategy_asset_log_returns)**(1/number_of_years) - 1
    sharpe_ratio_MA= (MA_average_yearly_return-risk_free_rate)/cum_strategy_asset_log_returns.std(0)

    buy_and_hold_MCS=pd.DataFrame(1, index = Targetdata_2020_MCS.index, columns=Targetdata_2020_MCS.columns)
    buy_and_hold_log_returns=buy_and_hold_MCS * asset_log_returns
    cum_buy_and_hold_log_returns = buy_and_hold_log_returns.cumsum()
    last_value_cum_buy_and_hold_log_returnss=buy_and_hold_log_returns.sum()
    buy_and_hold_average_yearly_return = (1 + last_value_cum_buy_and_hold_log_returnss)**(1/number_of_years) - 1
    sharpe_ratio_buy_and_hold= (buy_and_hold_average_yearly_return-risk_free_rate)/cum_buy_and_hold_log_returns.std(0)

    BB_log_returns=BB_trading_pos_final_MCS* asset_log_returns
    cum_BB_log_returns = BB_log_returns.cumsum()
    last_value_cum_BB_log_returnss=BB_log_returns.sum()
    BB_average_yearly_return = (1 + last_value_cum_BB_log_returnss)**(1/number_of_years) - 1
    sharpe_ratio_BB= (BB_average_yearly_return-risk_free_rate)/cum_BB_log_returns.std(0)

    """
    fig = plt.figure(figsize=(16,9))
    ax1 = plt.subplot2grid(shape=(16, 9), loc=(0, 0), rowspan=9,colspan=16)
    ax1.plot(cum_strategy_asset_log_returns.loc[start_date:end_date, :], label='MA strategy')
    ax1.plot(cum_buy_and_hold_log_returns.loc[start_date:end_date, :], label='Buy and hold')
    ax1.plot(cum_BB_log_returns.loc[start_date:end_date, :], label='BB strategy')
    ax1.title.set_text('Cumulative log-returns using MCS 20MA/BB/BNH in 2020 (Without Trading Cost)')
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
    """

    #Compute and Plot log return post cost of 2020 MA/BB/BNH strategy
    Targetdata_2020_BNH_MCS=cum_buy_and_hold_log_returns.copy(deep=True)
    Targetdata_2020_BNH_MCS=(1+Targetdata_2020_BNH_MCS)*1000000
    Targetdata_2020_BNH_MCS.iloc[1][0]=1000000

    Targetdata_2020_MA_year_return=(Targetdata_2020_MA_MCS.iloc[-1][6]/Targetdata_2020_MA_MCS.iloc[0][6])-1
    Targetdata_2020_BB_year_return=(Targetdata_2020_BB_MCS.iloc[-1][6]/Targetdata_2020_BB_MCS.iloc[0][6])-1
    Targetdata_2020_BNH_year_return=(Targetdata_2020_BNH_MCS.iloc[-1][0]/Targetdata_2020_BNH_MCS.iloc[1][0])-1

    """
    fig = plt.figure(figsize=(16,9))
    ax1 = plt.subplot2grid(shape=(16, 9), loc=(0, 0), rowspan=9,colspan=16)
    ax1.plot(Targetdata_2020_MA_MCS.loc[start_date:end_date,'Current_Value' :], label='MA strategy')
    ax1.plot(Targetdata_2020_BNH_MCS.loc[start_date:end_date,:], label='Buy and hold')
    ax1.plot(Targetdata_2020_BB_MCS.loc[start_date:end_date,'Current_Value' :], label='BB strategy')
    ax1.title.set_text('Assets Value using 20MA/BB/BNH in 2020 (With Trading Cost)')
    ax1.set_ylabel('Assets Value')
    ax1.legend(loc='best')

    ax2 = plt.subplot2grid(shape=(8, 5), loc=(6, 0), rowspan=1)
    column_labels=["Yearly Return"]
    row_labels=["20MA","BB","Buy and Hold"]
    ax2.title.set_text('Yearly return of different strategy in 2020 With Trading Cost (%)')
    data=[[100*np.float(Targetdata_2020_MA_year_return)],
          [100*np.float(Targetdata_2020_BB_year_return)],
          [100*np.float(Targetdata_2020_BNH_year_return)]]
    ax2.table(cellText=data,colLabels=column_labels,rowLabels=row_labels,loc='center')
    ax2.axis('tight')
    ax2.axis('off')

    ax3 = plt.subplot2grid(shape=(8, 5), loc=(6, 2), rowspan=1)
    column_labels=["Sharpe Ratio"]
    row_labels=["20MA","BB","Buy and Hold"]
    ax3.title.set_text('Sharpe Ratio of different strategy in 2020')
    data=[[Sharpe_ratio(Targetdata_2020_MA_year_return,Targetdata_2020_MA_MCS['Current_Value'])],
          [Sharpe_ratio(Targetdata_2020_BB_year_return,Targetdata_2020_BB_MCS['Current_Value'])],
          [Sharpe_ratio(Targetdata_2020_BNH_year_return,Targetdata_2020_BNH_MCS.iloc[:,0])]]
    ax3.table(cellText=data,colLabels=column_labels,rowLabels=row_labels,loc='center')
    ax3.axis('tight')
    ax3.axis('off')
    plt.tight_layout()
    plt.show()
    """

    
    Each_trial_post_cost_MA_return.append(Targetdata_2020_MA_year_return)
    Each_trial_post_cost_BB_return.append(Targetdata_2020_BB_year_return)
    Each_trial_post_cost_BNH_return.append(Targetdata_2020_BNH_year_return)

#Calculating the MCS post cost stat (MCS)
MCS_post_cost_MA_return_mean=np.mean(Each_trial_post_cost_MA_return)
MCS_post_cost_BB_return_mean=np.mean(Each_trial_post_cost_BB_return)
MCS_post_cost_BNH_return_mean=np.mean(Each_trial_post_cost_BNH_return)

MCS_post_cost_MA_SharRat=Sharpe_ratio(MCS_post_cost_MA_return_mean,Each_trial_post_cost_MA_return)
MCS_post_cost_BB_SharRat=Sharpe_ratio(MCS_post_cost_BB_return_mean,Each_trial_post_cost_BB_return)
MCS_post_cost_BNH_SharRat=Sharpe_ratio(MCS_post_cost_BNH_return_mean,Each_trial_post_cost_BNH_return)

fig = plt.figure(figsize=(16,9))
ax4 = plt.subplot2grid(shape=(8, 5), loc=(3, 2), rowspan=1)
column_labels=["Yearly Return"]
row_labels=["20MA","BB","Buy and Hold"]
ax4.title.set_text('Yearly return of all MCS 20MA/BB/BNH in 2020 With Trading Cost (%)')
data=[[100*np.float(MCS_post_cost_MA_return_mean)],
        [100*np.float(MCS_post_cost_BB_return_mean)],
        [100*np.float(MCS_post_cost_BNH_return_mean)]]
ax4.table(cellText=data,colLabels=column_labels,rowLabels=row_labels,loc='center')
ax4.axis('tight')
ax4.axis('off')

ax5 = plt.subplot2grid(shape=(8, 5), loc=(6, 2), rowspan=1)
column_labels=["Sharpe Ratio"]
row_labels=["20MA","BB","Buy and Hold"]
ax5.title.set_text('Sharpe Ratio of all MCS 20MA/BB/BNH in 2020 With Trading Cost')
data=[[MCS_post_cost_MA_SharRat],
      [MCS_post_cost_BB_SharRat],
      [MCS_post_cost_BNH_SharRat]]
ax5.table(cellText=data,colLabels=column_labels,rowLabels=row_labels,loc='center')
ax5.axis('tight')
ax5.axis('off')
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
