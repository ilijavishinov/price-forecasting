import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller, pacf
from typing import List, Tuple
from utils.io import *
from utils.metrics import *

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


def pqdm_grid(P, D, Q, m):
    pqdm = []
    for i in P:
        for j in D:
            for k in Q:
                for l in m:
                    pqdm.append((i, j, k, l))
    return pqdm


def adf_test_statistic(time_series):
    """
    """
    # set name
    adf_test = dict()
    
    # get return values
    adf_test['adfstat'], adf_test['pvalue'], adf_test['usedlag'], adf_test['nobs'], adf_test['critvalues'], adf_test['icbest'] = adfuller(time_series.dropna())
    
    adf_test['test_significant'] = adf_test['pvalue'] < 0.05
    
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
    


class TimeSeriesHandler(object):
    """
    """
    best_order = None
    forecast = None
    exog_forecast = None
    all_orders = None
    all_seasonal_orders = [None]
    
    def __init__(self,
                 name: str,
                 df: pd.DataFrame,
                 endog: str,
                 metadata_folder_suffix: str,
                 exog: List[str] = None,
                 endog_is_exog: bool = False):


        self.name = name
        self.df = df
        self.endog = endog
        self.exog = exog
        self.endog_is_exog = endog_is_exog

        # hardcoded train test split
        self.train = self.df.loc[:'2022-12', :]
        self.test = self.df.loc['2023-01':, :]
        
        # metadata dirs
        self.exog_suffix = '__withExog' if self.exog is not None else ''
        self.forecast_plots_dir = ensure_dir_exists(os.path.join(PLOTS_DIR, f'forecast{metadata_folder_suffix}', f'{self.name}__{self.endog}{self.exog_suffix}'))
        self.forecast_best_plots_dir = ensure_dir_exists(os.path.join(PLOTS_DIR, f'forecast_best{metadata_folder_suffix}'))
        self.models_dir = ensure_dir_exists('models')
        self.results_dir = ensure_dir_exists(f'results{metadata_folder_suffix}')

    
    def set_exog_forecast(self,
                          exog_forecast_df: pd.DataFrame):
        """
        Set 12-month forecast for the exogenous features
        """
        self.exog_forecast = exog_forecast_df
    
    def determine_diff_order(self):
        """
        Determine the differencing order of the series needed to make it stationary for ARIMA modeling
        """
        
        diff_ts = self.train[self.endog]
        
        diff_order = 0
        ts_stationary = False
        
        # calculate ts differences while ts is not stationary
        while not ts_stationary:
            time_series = diff_ts.diff().dropna()
            diff_order += 1
            ts_stationary = adf_test_statistic(time_series)['test_significant']
            
            # don't go above 3rd order differencing, as the series may never become stationary
            if diff_order == 3:
                break
        
        return diff_ts, diff_order
    
    def acf_lags(self):
        """
        Compute the autocorrelation function for the lags
        Return the significant lags which have corr coef above 0.2
        """
        ACF, ACF_ci = acf(self.train[self.endog], alpha = 0.05)
        lags = np.where(abs(ACF) > 0.2)[0]
        return lags
    
    def pacf_lags(self):
        """
        Compute the partial autocorrelation function for the lags
        Return the significant lags which have corr coef above 0.2
        """
        PACF, PACF_ci = pacf(self.train[self.endog], alpha = 0.05)
        lags = np.where(abs(PACF) > 0.2)[0]
        return lags
    
    def best_aic_order(self,
                       pacf_lags: List[int],
                       acf_lags: List[int],
                       diffs: List[int],
                       seasonal_orders: bool):
        """
        Get the best order depending on AIC statistic on the train set
        
        Currently returning the gred of all possible orders
        Currently not calculationg best orders based on AIC and BIC statistics (commented code)
        """
        pdqs = pdq_grid(p = pacf_lags if len(pacf_lags) > 0 else [0],
                        d = diffs,
                        q = acf_lags if len(acf_lags) > 0 else [0])
        
        pdqs_seasonal = pqdm_grid([0, 1], [0, 1, 2], [0, 1], [12]) if seasonal_orders else [None]
        
        return None, pdqs, pdqs_seasonal
        
        # currently not calculationg best orders based on AIC and BIC statistics
        
        # ic_list = list()
        # for pdq in pdqs:
        #     # print(f'Fitting {pdq} for {self.name}')
        #     model = SARIMAX(endog = self.train[self.endog],
        #                        exog = self.train[self.exog] if self.exog is not None else None,
        #                        order = pdq)
        #     model_fit = model.fit()
        #
        #     ic_list.append(dict(order = pdq,
        #                         aic = model_fit.aic,
        #                         bic = model_fit.bic))
        #
        # ic_df = pd.DataFrame(ic_list).sort_values('aic')
        # ensure_dir_exists('results')
        # ic_df.to_csv(f'results/train_aic_{self.name}.csv', index = False)
        # best_order = ic_df.iloc[0]['order']
        # return best_order, pdqs
    
    def plot_acf_pacf(self):
        """
        Plot the ACF and PACF functino values for the lags
        """
        ensure_dir_exists(os.path.join('plots', 'corr_funcs'))
        
        sm.graphics.tsa.plot_acf(self.df[self.endog], lags = 25)
        plt.savefig(os.path.join('plots', 'corr_funcs', f'{self.name}_{self.endog}_acf.pdf'))
        plt.close()
        
        sm.graphics.tsa.plot_pacf(self.df[self.endog], lags = 25)
        plt.savefig(os.path.join('plots', 'corr_funcs', f'{self.name}_{self.endog}_pacf.pdf'))
        plt.close()
    
    def determine_orders(self,
                         seasonal_orders: bool = False):
        """
        Determining the best orders for the current time series
        """
        diff_ts, order_diff = self.determine_diff_order()
        acf_lags = self.acf_lags()
        pacf_lags = self.pacf_lags()
        
        self.best_order, self.all_orders, self.all_seasonal_orders = self.best_aic_order(pacf_lags = pacf_lags,
                                                                                         acf_lags = acf_lags,
                                                                                         diffs = [order_diff],
                                                                                         seasonal_orders = seasonal_orders)
    
    def plot_forecast(self,
                      fit: pd.Series,
                      fit_lower: pd.Series,
                      fit_upper: pd.Series,
                      forecast: pd.Series,
                      plot_path: str,
                      order: Tuple,
                      seasonal_order: Tuple,
                      metrics: dict):
        """
        Plot ground truth, model fit and forecast with confidence bounds
        """
        fig, ax = plt.subplots(figsize = (10, 4))

        # ground truth plot
        ax.plot(self.df[self.endog], label = 'Ground Truth')
        
        # training model fit plot
        ax.plot(fit, label = 'Fit')
        ax.fill_between(x = self.train.index, y1 = fit_lower, y2 = fit_upper, color = 'orange', alpha = 0.2)
        
        # forecast plot
        ax.plot(forecast['mean'], label = 'Forecast')
        ax.fill_between(forecast.index, forecast['mean_ci_lower'], forecast['mean_ci_upper'], alpha = 0.2)

        # yearly vlines
        for i in vlines(self.df):
            ax.axvline(x = i, color = "black", alpha = 0.2, linestyle = "--")

        ax.legend()
        ax.set_title(f'Model fit {self.name}\n aic={round(metrics["aic"], 2)} bic={round(metrics["bic"], 2)} mape={round(metrics["mape"], 2)} order={order}, seasonal_order={seasonal_order}')
        plt.savefig(plot_path)
        plt.close()
        
    
    def fit_single_sarima(self,
                          model_name: str,
                          order: Tuple[int],
                          seasonal_order: Tuple[int]):
        """
        fit and save a SARIMAX model, or load it if it exists
        """
        
        if os.path.exists(os.path.join(self.models_dir, f'{model_name}.pkl')):
            model_fit = SARIMAXResults.load(os.path.join(self.models_dir, f'{model_name}.pkl'))
        else:
            model = SARIMAX(endog = self.train[self.endog],
                            exog = self.train[self.exog] if self.exog is not None else None,
                            order = order,
                            sesonal_order = seasonal_order)
            
            model_fit = model.fit(disp = 0)
            model_fit.save(os.path.join(self.models_dir, f'{model_name}.pkl'))

        return model_fit
    
    def forecast_single_sarima(self,
                               model_fit: SARIMAXResults):
        """
        Forecast the next X (the amount of test points) steps in the time series and calculate metrics
        """
        
        fcast = model_fit.get_forecast(steps = len(self.test), exog = self.exog_forecast).summary_frame()
        metrics = regression_metrics(self.test[self.endog], fcast['mean'])
        return fcast, metrics
    
    def fit_grid_orders_and_seasonal_orders(self,
                                            orders: List[Tuple],
                                            seasonal_orders: List[Tuple],
                                            best_metrics: List[str] = None,
                                            save_results_df: bool = False,
                                            plots: bool = True):
        
        """
        Run the grid of order and seasonal order combinations
        For each fit a SARIMAX model, forecast, calculate forecasting metrics, and make plots
        """
        
        if best_metrics is None:
            order_seasonal_order_metric_pairs = [[o, so, None] for o in orders for so in seasonal_orders]
        else:
            order_seasonal_order_metric_pairs = list(zip(orders, seasonal_orders, best_metrics))
            
        results_df = list()
        # for seasonal_order in seasonal_orders:
        #     for order in orders:
        for order, seasonal_order, best_metric in order_seasonal_order_metric_pairs:
            seasonal_suffix = '' if seasonal_order is None else f'_{str(seasonal_order)}'
            model_name = f'{self.name}__{self.endog}{self.exog_suffix}__{order}{seasonal_suffix}'
            
            try:
                model_fit = self.fit_single_sarima(model_name = model_name,
                                                   order = order,
                                                   seasonal_order = seasonal_order)
            except:
                print(f'Cannot fit {model_name}')
                continue
            
            # train data model fit with confidence interval
            pred_summary = model_fit.get_prediction(self.train.index[0], self.train.index[-1]).summary_frame()
            pred_list, y_lower, y_upper = pred_summary['mean'], pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper']
            
            # TODO: align series
            pred_list.index = pred_list.index - pd.Timedelta(days = 30.5)
            y_lower.index = y_lower.index - pd.Timedelta(days = 30.5)
            y_upper.index = y_upper.index - pd.Timedelta(days = 30.5)
            
            try:
                fcast, metrics = self.forecast_single_sarima(model_fit = model_fit)
                if best_metric == 'aic':
                    self.forecast = fcast['mean']
                metrics = dict(name = self.name, endog = self.endog,
                               order = str(order), seasonal_order = str(seasonal_order),
                               aic = model_fit.aic, bic = model_fit.bic,
                               **metrics)
                results_df.append(metrics)
            except:
                print(f'Cannot forecast {model_name}')
                continue
            
            if not self.endog_is_exog:
                model_name = f'{self.name}__{self.endog}{self.exog_suffix}__{order}{seasonal_suffix}'
            else: model_name = f'{self.name}__{order}{seasonal_suffix}'
            
            if best_metric is None:
                plot_path = f'{self.forecast_plots_dir}/{model_name}.jpg'
                plots = False
            else:
                plot_path = f'{self.forecast_best_plots_dir}/{model_name}_{best_metric}Best.jpg'
                
            if plots and not os.path.exists(plot_path):
                self.plot_forecast(fit = pred_list,
                                   fit_lower = y_lower,
                                   fit_upper = y_upper,
                                   forecast = fcast,
                                   plot_path = plot_path,
                                   order = order,
                                   seasonal_order = seasonal_order,
                                   metrics = metrics)
        
        # save summary df for forecast with all orders
        results_df = pd.DataFrame(results_df)
        if save_results_df:
            seasonal_suffix = '' if seasonal_order is None else '_withSeason'
            results_df.to_csv(os.path.join(self.results_dir, f'{self.name}__{self.endog}{self.exog_suffix}__all{seasonal_suffix}.csv'), index = False)
        return results_df
    
    def fit_and_forecast_sarimax(self,
                                 all_orders: bool = False):
        """
        Run the grid of set orders and seasonal orders, choose the best orders depending on AIC nad MAPE
        For the best orders, save plots and results in a separate folder
        """

        # exit if results have been calculated
        # if os.path.exists(f'results/{self.name}__{self.endog}{exog_suffix}__all.csv'):
        #   return
        
        results_df = self.fit_grid_orders_and_seasonal_orders(
            orders = [self.best_order] if not all_orders else self.all_orders,
            seasonal_orders = self.all_seasonal_orders,
            save_results_df = True
        )
        
        if len(results_df) > 1:
            best_order_aic = eval(results_df.loc[results_df['aic'].idxmin()]['order'])
            best_order_mape = eval(results_df.loc[results_df['mape'].idxmin()]['order'])
            
            best_order_seasonal_aic = eval(results_df.loc[results_df['aic'].idxmin()]['seasonal_order'])
            best_order_seasonal_mape = eval(results_df.loc[results_df['mape'].idxmin()]['seasonal_order'])
        
            
            self.fit_grid_orders_and_seasonal_orders(orders = [best_order_aic, best_order_mape],
                                                     seasonal_orders = [best_order_seasonal_aic, best_order_seasonal_mape],
                                                     best_metrics = ['aic', 'mape'])
                























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




