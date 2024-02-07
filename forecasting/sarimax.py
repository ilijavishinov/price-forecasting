exog_features = [
    'MGDP_Index',
    'EBIT_Index',
    'Revenue_Index',
    'fleet1',
    'fleet2',
    'fleet3',
    'Copper',
    'Gold',
    'Iron Ore',
    'Nickel',
    'Silver',
    'Zinc',
]

import importlib
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

import warnings

warnings.filterwarnings('ignore')

# %%
from utils import ts_data
from utils import ts_utils

importlib.reload(ts_data)
importlib.reload(ts_utils)

data_handler = ts_data.TsDataHandler(data_path = '../data')

bs_areas = data_handler.get_bussines_areas()
# bs_areas = ['EQ']
# bs_areas = ['Consumables']

regions = data_handler.get_regions()
# regions = [33]
# regions = [56]

all_models = True

forecasted_exogs_list = list()
forecasted_exogs = dict()

for bs_area in bs_areas:
    for region in regions:
        if bs_area == 'Consumables' and region == 56:
            continue
        name = f'{bs_area}_{region}'
        print(name)
        
        sales_bus_region = data_handler.get_sales_per_business_area_and_region(business_area = bs_area,
                                                                               region = region)
        
        sales_bus_region = data_handler.add_monthly_financial_info(sales_df = sales_bus_region,
                                                                   region = region)
        
        ts_handler = ts_utils.TimeSeriesHandler(name = name,
                                                df = sales_bus_region,
                                                endog = 'sales_value')
                                                # endog = 'sales_value', exog = exog_features)
        
        if len(ts_handler.test) < 5:
            print('TEST NOT FULL ' + name)
            print(len(ts_handler.test))
        
        # exog_forecast = pd.DataFrame()
        # for exog in ts_handler.exog:
        #     exog_already_forecasted = False
        #
        #     exog_id = str(sales_bus_region[exog].values.tolist())
        #
        #     # check if current time series has already been forecasted
        #     for forecasted_exog in forecasted_exogs:
        #         if exog_id in forecasted_exogs.keys():
        #             exog_already_forecasted = True
        #
        #     if exog_already_forecasted:
        #         exog_forecast[exog] = forecasted_exogs[exog_id]
        #         print(exog, ' already forecasted')
        #         continue
        #     else:
        #         print(exog)
        #         exog_name = f'exog_{exog}__{name}'
        #
        #         exog_ts_handler = ts_utils.TimeSeriesHandler(name = exog_name,
        #                                                      df = sales_bus_region,
        #                                                      endog = exog,
        #                                                      exog = None,
        #                                                      endog_is_exog = True)
        #         if len(exog_ts_handler.test) < 5:
        #             print('TEST NOT FULL ' + name + ' ' + exog_name)
        #             print(len(exog_ts_handler.test))
        #
        #         exog_ts_handler.determine_orders()
        #         # exog_ts_handler.fit_sarimax_exog(plot = False if all_models else True)
        #         exog_ts_handler.fit_sarimax(all_orders = True)
        #         exog_forecast[exog] = exog_ts_handler.forecast
        #         forecasted_exogs[exog_id] = exog_ts_handler.forecast
        
        # ts_handler.set_exog_forecast(exog_forecast)
        
        # ts_handler.plot_acf_pacf()
        
        ts_handler.determine_orders()
        
        ts_handler.fit_sarimax(all_orders = True)
        
        if not all_models: break
    if not all_models: break



