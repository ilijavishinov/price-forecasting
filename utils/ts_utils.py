from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import sqrt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller, pacf
import statsmodels.api as sm
import os
from utils.metrics import *
from utils.io import *

PLOTS_DIR = 'plots'


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


def ts_decomposition(df,
                     endog,
                     name,
                     show_plot = False):
    """
    Season-Trend decomposition using LOESS
    """
    result = STL(endog = df[endog], period = 12, seasonal = 13, robust = True).fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    
    fig, axs = plt.subplots(nrows = 5, sharex = False, figsize = (15, 20))
    
    for ax in axs:
        for i in vlines(df):
            ax.axvline(x = i, color = "black", alpha = 0.2, linestyle = "--")
    
    axs[0].plot(df[endog])
    axs[0].set_title('Original')
    axs[1].plot(seasonal)
    axs[1].set_title('Seasonal')
    axs[2].plot(trend)
    axs[2].set_title('Trend')
    axs[3].plot(resid)
    axs[3].set_title('Residual')
    axs[4].plot(trend + seasonal, label = "Trend + Seasonal")
    axs[4].plot(df[endog], label = "Original")
    axs[4].legend()
    
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'season_trend_decomp'))
    plt.savefig(os.path.join(PLOTS_DIR, 'season_trend_decomp', f'{endog}_{name}.pdf'))
    
    if show_plot:
        plt.close(fig)
    
    plt.close(fig)
    
    
    
    # check where the null values are
    # df_null = df.isna()
    # df_null[df_null['sales_diff'] == True].index



class TimeSeriesHandler(object):
    """
    """
    df = None
    best_order = None
    endog = None
    exog = None
    forecast = None
    exog_forecast = None
    seasonal_order = None
    all_orders = None
    
    def __init__(self,
                 name,
                 df,
                 endog,
                 exog = None,
                 endog_is_exog = False):
        
        self.name = name
        self.df = df
        self.endog = endog
        self.exog = exog
        
        self.train = self.df.loc[:'2022-12', :]
        self.test = self.df.loc['2023-01':, :]
        
        self.endog_is_exog = endog_is_exog
    
    
    def set_exog_forecast(self,
                          exog_forecast_df):
        self.exog_forecast = exog_forecast_df
    
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
            # print(f'{diff_order} stationary {ts_stationary}')
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
                        q = acf_lags if len(acf_lags) > 0 else [0])
        
        if not self.endog_is_exog:
            return (0, 0, 0), pdqs
        else:
            return (0, 0, 0), pdqs
            
            # ic_list = list()
            # for pdq in pdqs:
            #     # print(f'Fitting {pdq} for {self.name}')
            #     ar_model = SARIMAX(endog = self.train[self.endog],
            #                        exog = self.train[self.exog] if self.exog is not None else None,
            #                        order = pdq)
            #     ar_model_fit = ar_model.fit()
            #
            #     ic_list.append(dict(order = pdq,
            #                         aic = ar_model_fit.aic,
            #                         bic = ar_model_fit.bic))
            #
            # ic_df = pd.DataFrame(ic_list).sort_values('aic')
            # ensure_dir_exists('results')
            # ic_df.to_csv(f'results/train_aic_{self.name}.csv', index = False)
            # best_order = ic_df.iloc[0]['order']
            # return best_order, pdqs
    
    def plot_acf_pacf(self):
        
        ensure_dir_exists(os.path.join('plots', 'corr_funcs'))
        
        sm.graphics.tsa.plot_acf(self.df[self.endog], lags = 25)
        plt.savefig(os.path.join('plots', 'corr_funcs', f'{self.name}_{self.endog}_acf.pdf'))
        plt.close()
        
        sm.graphics.tsa.plot_pacf(self.df[self.endog], lags = 25)
        plt.savefig(os.path.join('plots', 'corr_funcs', f'{self.name}_{self.endog}_pacf.pdf'))
        plt.close()
    
    def determine_orders(self):
        
        diff_ts, order_diff = self.determine_diff_order()
        acf_lags = self.acf_lags()
        pacf_lags = self.pacf_lags()
        
        self.best_order, self.all_orders = self.best_aic_order(pacf_lags = pacf_lags,
                                                               acf_lags = acf_lags,
                                                               diffs = [order_diff])
    
    def fit_sarimax_exog(self,
                         plot = False):
        
        
        model = SARIMAX(endog = self.train[self.endog],
                        exog = self.train[self.exog] if self.exog is not None else None,
                        order = self.best_order)
        
        model_fit = model.fit()
        pred_summary = model_fit.get_prediction(self.train.index[0], self.train.index[-1]).summary_frame()
        pred_list = pred_summary['mean']
        y_lower = pred_summary['mean_ci_lower']
        y_upper = pred_summary['mean_ci_upper']
        
        pred_list.index = pred_list.index - pd.Timedelta(days = 30.5)
        y_lower.index = y_lower.index - pd.Timedelta(days = 30.5)
        y_upper.index = y_upper.index - pd.Timedelta(days = 30.5)
        
        fcast = model_fit.get_forecast(steps = len(self.test)).summary_frame()
        self.forecast = fcast['mean']
        
        if plot:
            fig, ax = plt.subplots(figsize = (20, 5))
            ax.plot(self.df[self.endog], label = 'Ground Truth')
            ax.plot(pred_list, label = 'Fit')
            # ax.fill_between(x = self.train.index, y1 = y_lower, y2 = y_upper, color = 'orange', alpha = 0.2)
            ax.plot(fcast['mean'], label = 'Forecast')
            # ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], alpha = 0.2)
            
            for i in vlines(self.df):
                ax.axvline(x = i, color = "black", alpha = 0.2, linestyle = "--")
            ax.legend()
            ax.set_title(f'Model fit {self.name} \n order {self.best_order}')
            plt.savefig(f'plots/{self.name}')
            fig.show()
    
    
    
    def fit_sarimax(self,
                    seasonal_order = None,
                    all_orders = False,
                    plots = True):
        
        # whether forecasting exog
        if self.exog is not None: exog_suffix = '__withExog'
        else: exog_suffix = ''
        
        # metadata dirs
        forecast_plots_dir = ensure_dir_exists(os.path.join(PLOTS_DIR, 'forecast', f'{self.name}__{self.endog}{exog_suffix}'))
        models_dir = ensure_dir_exists('models')
        results_dir = ensure_dir_exists('results')
        
        # exit if results have been calculated
        # if os.path.exists(f'results/{self.name}__{self.endog}{exog_suffix}__all.csv'): return
        
        iterating_orders = [self.best_order] if not all_orders else self.all_orders
        results_df = list()
        for order in iterating_orders:
            model_name = f'{self.name}__{self.endog}{exog_suffix}__{order}'
            
            # model_fit = None
            
            if os.path.exists(os.path.join(models_dir, f'{model_name}.pkl')):
                # with open(os.path.exists(os.path.join(models_dir, f'{model_name}.pkl')), 'rb') as file:
                model_fit = SARIMAXResults.load(os.path.join(models_dir, f'{model_name}.pkl'))
            else:
                # if Linear Algebra error, go to next order
                try:
                    model = SARIMAX(endog = self.train[self.endog],
                                    exog = self.train[self.exog] if self.exog is not None else None,
                                    order = order,
                                    sesonal_order = seasonal_order)
                    
                    model_fit = model.fit(disp=0)
                    model_fit.save(os.path.join(models_dir, f'{model_name}.pkl'))
                except:
                    print(f'------------ CANNOT FIT {model_name}')
                    continue
            
            # train data model fit with confidence interval
            pred_summary = model_fit.get_prediction(self.train.index[0], self.train.index[-1]).summary_frame()
            pred_list, y_lower, y_upper = pred_summary['mean'], pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper']
            
            # TODO: align series
            pred_list.index = pred_list.index - pd.Timedelta(days = 30.5)
            y_lower.index = y_lower.index - pd.Timedelta(days = 30.5)
            y_upper.index = y_upper.index - pd.Timedelta(days = 30.5)

            try:
                # forecast
                fcast = model_fit.get_forecast(steps = len(self.test), exog = self.exog_forecast).summary_frame()
            except:
                print(f'------------ CANNOT FORECAST {model_name}')
                continue
            
            # forecast metrics calculate and save
            metrics = regression_metrics(self.test[self.endog], fcast['mean'])
            metrics = dict(
                name = self.name,
                endog = self.endog,
                order = str(order),
                aic = model_fit.aic,
                bic = model_fit.bic,
                **metrics
            )
            results_df.append(metrics)
            # save_json_metadata(metadata_dir = 'results/separate',
            #                    file_name = f'{model_name}',
            #                    data = metrics)
            
            
            # show plots
            if plots:
                if not os.path.exists(f'{forecast_plots_dir}/{model_name}.jpg'):
                    fig, ax = plt.subplots(figsize = (10, 4))
                    
                    ax.plot(self.df[self.endog], label = 'Ground Truth')
                    ax.plot(pred_list, label = 'Fit')
                    ax.fill_between(x = self.train.index, y1 = y_lower, y2 = y_upper, color = 'orange', alpha = 0.2)
                    
                    ax.plot(fcast['mean'], label = 'Forecast')
                    ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], alpha = 0.2)
                    
                    # yearly vlines
                    for i in vlines(self.df):
                        ax.axvline(x = i, color = "black", alpha = 0.2, linestyle = "--")
                    # legend and title
                    ax.legend()
                    ax.set_title(f'Model fit {self.name}\n mape={round(metrics["mape"], 2)} order={order}, seasonal_order={self.seasonal_order}')
                    # save and close
                    plt.savefig(f'{forecast_plots_dir}/{model_name}.jpg')
                    plt.close()
        
        # save summary df for forecast with all orders
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(os.path.join(results_dir, f'{self.name}__{self.endog}{exog_suffix}__all.csv'), index = False)
        
        
        # choose best aic model
        try:
            best_order_aic = eval(results_df.loc[results_df['aic'].idxmin()]['order'])
            best_order_mape = eval(results_df.loc[results_df['mape'].idxmin()]['order'])
            
        except:
            model_name = f'{self.name}__{self.endog}{exog_suffix}'
            forecast_plots_dir = ensure_dir_exists(os.path.join(PLOTS_DIR, 'forecast_best'))
            fig, ax = plt.subplots(figsize = (10, 4))
            plt.savefig(f'{forecast_plots_dir}/{model_name}.jpg')
            return
        
        for metric_name, best_order_aic_mape in zip(['aic', 'mape'], [best_order_aic, best_order_mape]):
            model_name = f'{self.name}__{self.endog}{exog_suffix}__{best_order_aic_mape}'
            forecast_plots_dir = ensure_dir_exists(os.path.join(PLOTS_DIR, 'forecast_best'))
    
            model = SARIMAX(endog = self.train[self.endog],
                            exog = self.train[self.exog] if self.exog is not None else None,
                            order = best_order_aic_mape,
                            sesonal_order = seasonal_order)
            
            model_fit = model.fit(disp=0)
            
            # train data model fit with confidence interval
            pred_summary = model_fit.get_prediction(self.train.index[0], self.train.index[-1]).summary_frame()
            pred_list, y_lower, y_upper = pred_summary['mean'], pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper']
            
            # TODO: align series
            pred_list.index = pred_list.index - pd.Timedelta(days = 30.5)
            y_lower.index = y_lower.index - pd.Timedelta(days = 30.5)
            y_upper.index = y_upper.index - pd.Timedelta(days = 30.5)
            
            fcast = model_fit.get_forecast(steps = len(self.test), exog = self.exog_forecast).summary_frame()
            self.forecast = fcast['mean']
            
            # forecast metrics calculate and save
            metrics = regression_metrics(self.test[self.endog], fcast['mean'])
            
            
            # show plots
            if plots:
                fig, ax = plt.subplots(figsize = (10, 4))
                
                ax.plot(self.df[self.endog], label = 'Ground Truth')
                ax.plot(pred_list, label = 'Fit')
                ax.fill_between(x = self.train.index, y1 = y_lower, y2 = y_upper, color = 'orange', alpha = 0.2)
                
                ax.plot(fcast['mean'], label = 'Forecast')
                ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], alpha = 0.2)
                
                # yearly vlines
                for i in vlines(self.df):
                    ax.axvline(x = i, color = "black", alpha = 0.2, linestyle = "--")
                # legend and title
                ax.legend()
                ax.set_title(f'Model fit {self.name}\n mape={round(metrics["mape"], 2)} order={best_order_aic_mape}, seasonal_order={self.seasonal_order}')
                # save and close
                plt.savefig(f'{forecast_plots_dir}/{model_name}_{metric_name}Best.jpg')
                plt.close()

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




