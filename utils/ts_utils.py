import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller, pacf


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
    

def pdq_grid(p, d, q):
    grid = []
    for i in p:
        for j in d:
            for k in q:
                grid.append((i, j, k))
    return grid


def adf_summary(time_series):
    """
    """
    # set name
    adf_test = dict()
    
    # get return values
    adf_test['adfstat'], adf_test['pvalue'], adf_test['usedlag'], adf_test['nobs'], adf_test['critvalues'], adf_test['icbest'] = adfuller(time_series.dropna())
    
    adf_test['test_significant'] = adf_test['pvalue'] < 0.25
    
    # unpack critical values
    for pct in [1, 5, 10]:
        adf_test[f'critical_value_{pct}pct'] = adf_test['critvalues'][f'{pct}%']
        adf_test[f'critical_value_{pct}pct_significant'] = adf_test['adfstat'] < adf_test[f'critical_value_{pct}pct']
    
    adf_test.pop('critvalues')
    
    return adf_test

class TimeSeriesHandler(object):
    """
    """
    df = None
    best_order = None
    endog = None
    exog = None
    forecast = None
    
    def __init__(self,
                 name,
                 df,
                 endog,
                 exog = None):
        
        self.name = name
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
            print(f'{diff_order} stationary {ts_stationary}')
            if diff_order == 3:
                break
        
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
        for pdq in pdqs[:3]:
            print(f'Fitting {pdq} for {self.name}')
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
        
    def fit_sarimax_exog(self,
                         plot = False):
        
        fig, ax = plt.subplots(figsize = (15, 5))
        
        model = SARIMAX(endog = self.train[self.endog],
                        exog = self.train[self.exog] if self.exog is not None else None,
                        order = self.best_order)
        
        model_fit = model.fit()
        pred_summary = model_fit.get_prediction(self.train.index[0], self.train.index[-1]).summary_frame()
        pred_list = pred_summary['mean']
        y_lower = pred_summary['mean_ci_lower']
        y_upper = pred_summary['mean_ci_upper']
        
        fcast = model_fit.get_forecast(steps = len(self.test)).summary_frame()
        self.forecast = fcast
        
        if plot:
            ax.plot(self.df[self.endog], label = 'Ground Truth')
            ax.plot(pred_list, label = 'Fit')
            # ax.fill_between(x = self.train.index, y1 = y_lower, y2 = y_upper, color = 'orange', alpha = 0.2)
            ax.plot(fcast['mean'], label = 'Forecast')
            # ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], alpha = 0.2)
            
            for i in vlines(self.df):
                ax.axvline(x = i, color = "black", alpha = 0.2, linestyle = "--")
            ax.legend()
            ax.set_title(f'Model fit {self.name}')
            fig.show()
            
        
        
    def fit_sarimax(self,
                    name,
                    plot_fit = False,
                    plot_prediction = False,
                    rolling_prediction = False,
                    plot_rolling = False):
        
        fig, ax = plt.subplots(figsize = (15, 5))
        show_plot = any([plot_fit, plot_prediction, plot_rolling])
        
        model = SARIMAX(endog = self.train[self.endog],
                        exog = self.train[self.exog] if self.exog is not None else None,
                        order = self.best_order)
        
        model_fit = model.fit()
        pred_summary = model_fit.get_prediction(self.train.index[0], self.train.index[-1]).summary_frame()
        pred_list = pred_summary['mean']
        y_lower = pred_summary['mean_ci_lower']
        y_upper = pred_summary['mean_ci_upper']
        
        # pred_list.index = pred_list.index - pd.Timedelta(days = 30.5)
        # y_lower.index = y_lower.index - pd.Timedelta(days = 30.5)
        # y_upper.index = y_upper.index - pd.Timedelta(days = 30.5)
        
        if plot_fit:
            ax.plot(self.df[self.endog], label = 'Ground Truth')
            ax.plot(pred_list, label = 'Fit')
            ax.fill_between(x = self.train.index, y1 = y_lower, y2 = y_upper, color = 'orange', alpha = 0.2)
        
        # forecast
        fcast = model_fit.get_forecast(steps = len(self.test), exog = self.test[self.exog]).summary_frame()
        if plot_prediction:
            ax.plot(fcast['mean'], label = 'Forecast')
            ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], alpha = 0.2)
        
        # # rolling prediction
        # predictions = pd.Series()
        # history = train
        # for i in range(len(test)):
        #     model = SARIMAX(endog = history[self.endog],
        #                     exog = history[self.exog] if self.exog is not None else None,
        #                     order = self.best_order)
        #
        #     model_fit = model.fit()
        #     output = model_fit.forecast(steps = 1)
        #     predictions = pd.concat([predictions, output])
        #     history = pd.concat([history, test[i:i + 1]])
        #
        # if plot_rolling:
        #     ax.plot(predictions, label = 'Rolling Prediction')
        #     # ax.set_title(f'{self.name}\n rmse={round(rmse, 4)}\n order={self.best_order}, seasonal_order={seasonal_order}')
        
        if show_plot:
            for i in vlines(self.df):
                ax.axvline(x = i, color = "black", alpha = 0.2, linestyle = "--")
            ax.legend()
            ax.set_title(f'Model fit {name}')
            fig.show()
    
