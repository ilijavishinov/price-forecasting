from constants import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from utils import ts_data
from utils import ts_utils

DATA_PATH = '../data'

USE_SEASONAL_ORDERS = False
USE_EXOG = False
METADATA_FOLDER_SUFFIX = '_noShift'
if USE_SEASONAL_ORDERS: METADATA_FOLDER_SUFFIX += '_withSeason'
if USE_EXOG: METADATA_FOLDER_SUFFIX += '_withExog'


if __name__ == '__main__':
    
    data_handler = ts_data.TsDataHandler(data_path = DATA_PATH)
    
    forecasted_exogs_list = list()
    forecasted_exogs = dict()
    
    # iterate all region and business areas combinations
    for bs_area in data_handler.get_bussines_areas():
        for region in data_handler.get_regions():
            
            # exclude this combination
            if bs_area == 'Consumables' and region == 56:
                continue
                
            name = f'{bs_area}_{region}'
            print(name)
            
            # get sales for current region and business area
            sales_bus_region = data_handler.get_sales_per_business_area_and_region(business_area = bs_area,
                                                                                   region = region)
            
            # join exogenous features to current sales
            sales_bus_region = data_handler.add_monthly_financial_info(sales_df = sales_bus_region,
                                                                       region = region)
            
            # TimeSeriesHandler object for the sales timeseries
            ts_handler = ts_utils.TimeSeriesHandler(name = name,
                                                    df = sales_bus_region,
                                                    endog = 'sales_value',
                                                    metadata_folder_suffix = METADATA_FOLDER_SUFFIX,
                                                    exog = exog_features if USE_EXOG else None)
            
            # if using exogenous variables, forecast them too
            if USE_EXOG:

                exog_forecast = pd.DataFrame()
                for exog in ts_handler.exog:
                    
                    exog_already_forecasted = False
                    exog_id = str(sales_bus_region[exog].values.tolist())
        
                    # check if current time series has already been forecasted
                    for forecasted_exog in forecasted_exogs:
                        if exog_id in forecasted_exogs.keys():
                            exog_already_forecasted = True
                    
                    if exog_already_forecasted:
                        exog_forecast[exog] = forecasted_exogs[exog_id]
                        print(exog, ' already forecasted')
                        continue
                        
                    else:
                        print(exog, ' forecasting')
                        exog_ts_handler = ts_utils.TimeSeriesHandler(name = f'exog_{exog}__{name}',
                                                                     df = sales_bus_region,
                                                                     endog = exog,
                                                                     metadata_folder_suffix = METADATA_FOLDER_SUFFIX,
                                                                     exog = None,
                                                                     endog_is_exog = True)

                        # get best orders for the exog time series
                        exog_ts_handler.determine_orders()
                        
                        # fit the grid or the best orders for the exog time series
                        exog_ts_handler.fit_and_forecast_sarimax(all_orders = True)
                        
                        # save the exog forecast
                        exog_forecast[exog] = exog_ts_handler.forecast
                        forecasted_exogs[exog_id] = exog_ts_handler.forecast
                
                # add the foreasted exog features to the main sales time series
                ts_handler.set_exog_forecast(exog_forecast)
            
            # determine the best orders for forecasting sales
            ts_handler.determine_orders(seasonal_orders = USE_SEASONAL_ORDERS)
            
            # fit the grid or the best orders for the main sales time series
            ts_handler.fit_and_forecast_sarimax(all_orders = True)
            
    
    
