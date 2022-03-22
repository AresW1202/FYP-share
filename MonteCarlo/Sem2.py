import pandas as pd
from pandas_datareader import data
import numpy as np, numpy.random
from numpy import mean
import random
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm 
from scipy.stats import kstest
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats

def extract_prices(start_date,end_date,symbols,backtestduration=0):
    dim=len(symbols)
    for symbol in symbols:
        dfprices = data.DataReader(symbols, start=start_date, end=end_date, data_source='yahoo')
        dfprices = dfprices[['Adj Close']]
    dfprices.columns=[' '.join(col).strip() for col in dfprices.columns.values]
    for i in range(0,len(symbols)):
        noOfShares.append(portfolioValPerSymbol[i]/priceAtEndDate[i])
    noOfShares=[round(element, 5) for element in noOfShares]
    listOfColumns=dfprices.columns.tolist()   
    dfprices["Adj Close Portfolio"]=dfprices[listOfColumns].mul(noOfShares).sum(1)
    
    print(f"Extracted {len(dfprices)} days worth of data for {len(symbols)} counters with {dfprices.isnull().sum().sum()} missing data")
    
    return dfprices
  
  def calc_returns(dfprices,symbols):
    dfreturns=pd.DataFrame()
    columns = list(dfprices) 
    mean=[]
    stdev=[]
    for column in columns:
        dfreturns[f'Log Daily Returns {column}']=np.log(dfprices[column]).diff()
        mean.append(dfreturns[f'Log Daily Returns {column}'][1:].mean())
        stdev.append(dfreturns[f'Log Daily Returns {column}'][1:].std())
    dfreturns=dfreturns.dropna()
    
    if len(dfreturns.columns)==1:
        df_mean_stdev=pd.DataFrame(list(zip(symbols,mean,stdev)),columns =['Stock', 'Mean Log Daily Return','StdDev Log Daily Return']) 
    else:
        df_mean_stdev=pd.DataFrame(list(zip(symbols+["Portfolio"],mean,stdev)),columns =['Stock', 'Mean Log Daily Return','StdDev Log Daily Return'])
    
    return dfreturns ,df_mean_stdev
  
  def GBMsimulatorUniVar(So, mu, sigma, T, N):
    """
    Parameters
    So:     initial stocks' price
    mu:     expected return
    sigma:  volatility
    Cov:    covariance matrix
    T:      time period
    N:      number of increments
    """
    dim = np.size(So)
    t = np.linspace(0., T, int(N))
    S = np.zeros([dim, int(N)])
    S[:, 0] = So
    for i in range(1, int(N)):    
        drift = (mu - 0.5 * sigma**2) * (t[i] - t[i-1])
        Z = np.random.normal(0., 1., dim)
        diffusion = sigma* Z * (np.sqrt(t[i] - t[i-1]))
        S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
    return S, t
  
  def create_covar(dfreturns):  
    try:
        returns=[]
        arrOfReturns=[]
        columns = list(dfreturns)
        for column in columns:
            returns=dfreturns[column].values.tolist()
            arrOfReturns.append(returns)
        Cov = np.cov(np.array(arrOfReturns))    
        return Cov
    except LinAlgError :
        Cov = nearPD(np.array(arrOfReturns), nit=10)
        print("WARNING -Original Covariance Matrix is NOT Positive Semi Definite And Has Been Adjusted To Allow For Cholesky Decomposition ")
        return Cov
def GBMsimulatorMultiVar(So, mu, sigma, Cov, T, N):
    """
    Parameters
    So:     initial stocks' price
    mu:     expected return
    sigma:  volatility
    Cov:    covariance matrix
    T:      time period
    N:      number of increments
    """
    dim = np.size(So)
    t = np.linspace(0., T, int(N))
    A = np.linalg.cholesky(Cov)
    S = np.zeros([dim, int(N)])
    S[:, 0] = So
    for i in range(1, int(N)):    
        drift = (mu - 0.5 * sigma**2) * (t[i] - t[i-1])
        Z = np.random.normal(0., 1., dim)
        diffusion = np.matmul(A, Z) * (np.sqrt(t[i] - t[i-1]))
        S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
    return S, t
