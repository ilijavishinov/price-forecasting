import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
def rename_columns(columns,
                   rename_dict):
    """
    """
    for i in range(len(columns)):
        if columns[i] in rename_dict.keys():
            columns[i] = rename_dict[columns[i]]
    return columns


def vlines(df):
    """
    """
    years = df.index.strftime('%Y').unique().tolist()
    if len(years) > 0:
        years = years + [str(int(years[-1]) + 1)]
        dt_years = np.arange(years[0], years[-1], dtype = "datetime64[Y]")
        return dt_years
    else:
        return []



class TimeSeriesHandler(object):
    """
    """
    df = None
    best_order = None
    endog = None
    exog = None
    
    def __init__(self,
                 name,
                 df,
                 endog,
                 exog = None):
        
        self.name = name,
        self.df = df
        self.endog = endog
        self.exog = exog
        
        self.train = self.df.loc[:'2022-12', :]
        self.test = self.df.loc['2023-01':, :]
    
    def determine_diff_order(self):
        """
        """
        
        diff_ts = self.train[self.endog]
        
        diff_order = 0
        ts_stationary = False
        
        while not ts_stationary:
            time_series = diff_ts.diff().dropna()
            diff_order += 1
            ts_stationary = adf_summary(time_series)['test_significant']
        
        return diff_ts, diff_order
    
    def acf_lags(self):
        """
        """
        ACF, ACF_ci = acf(self.train[self.endog], alpha = 0.05)
        lags = np.where(abs(ACF) > 0.2)[0]
        return lags
    
    def pacf_lags(self):
        """
        """
        PACF, PACF_ci = pacf(self.train[self.endog], alpha = 0.05)
        lags = np.where(abs(PACF) > 0.2)[0]
        return lags
    
    def best_aic_order(self,
                       pacf_lags,
                       acf_lags,
                       diffs):
        """
        """
        pdqs = pdq_grid(p = pacf_lags if len(pacf_lags) > 0 else [0],
                        d = diffs,
                        q = acf_lags if len(acf_lags) > 0 else [0], )
        
        ic_list = list()
        for pdq in pdqs[:10]:
            
            ar_model = ARIMA(endog = self.train[self.endog],
                             exog = self.train[self.exog] if self.exog is not None else None,
                             order = pdq)
            ar_model_fit = ar_model.fit()
            
            ic_list.append(dict(order = pdq,
                                aic = ar_model_fit.aic,
                                bic = ar_model_fit.bic))
        
        ic_df = pd.DataFrame(ic_list).sort_values('aic')
        best_order = ic_df.iloc[0]['order']
        return best_order
    
    def determine_orders(self):
        
        diff_ts, order_diff = self.determine_diff_order()
        acf_lags = self.acf_lags()
        pacf_lags = self.pacf_lags()
        
        self.best_order = self.best_aic_order(pacf_lags = pacf_lags,
                                              acf_lags = acf_lags,
                                              diffs = [order_diff])
    
