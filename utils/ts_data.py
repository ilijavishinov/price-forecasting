import pandas as pd
from utils.io import *


class TsDataHandler(object):
    """
    Class that handles Ignize Time Series data:
        - Processes format to match ts format
        - Joins sales with MGDP, EBIT, Revenue, Commodities, Fleets
        - Has getters for data info
        
    """
    
    def __init__(self,
                 data_path):
        self.data_path = data_path
        
        self.comodity_prices_df = pd.read_csv(os.path.join(data_path, 'commodity_long.csv'))
        self.comodity_prices_df.index = pd.DatetimeIndex(self.comodity_prices_df['ts_month'])
        
        self.financial_index_df = pd.read_csv(os.path.join(data_path, 'comp fin index.csv'))
        self.financial_index_df.index = pd.DatetimeIndex(self.financial_index_df.date)
        self.financial_index_df.drop(['date', 'Unnamed: 0'], axis = 1, inplace = True)
        
        self.fleets_df = pd.read_csv(os.path.join(data_path, 'fleet feats test.csv'))
        self.fleets_df['date'] = pd.to_datetime(self.fleets_df['Delivery Date'])
        self.fleets_df.drop('Delivery Date', axis = 1, inplace = True)
        
        self.mgdp_df = pd.read_csv(os.path.join(data_path, 'mgdp index.csv'))
        self.mgdp_df.index = pd.DatetimeIndex(self.mgdp_df.date)
        self.mgdp_df.drop(['date', 'Unnamed: 0'], axis = 1, inplace = True)
        
        self.sales_df = pd.read_csv(os.path.join(data_path, 'test sales.csv'))
        self.sales_df['sales_date'] = pd.to_datetime(self.sales_df['sales_date'])
    
    def get_bussines_areas(self):
        """
        Get all business areas in the dataset
        """
        return list(set(self.sales_df['business_area_id'].unique().tolist()) - {'Other'})
    
    def get_regions(self):
        """
        Get all regions in the dataset
        """
        return self.sales_df['region_id'].unique().tolist()
    
    def get_sales_per_business_area_and_region(self,
                                               business_area,
                                               region):
        """
        Get sales data for specific business area and region
        """
        
        # select sales on current business area and region
        sales_df_barea_region = self.sales_df[(self.sales_df['region_id'] == region) & (self.sales_df['business_area_id'] == business_area)].sort_values(['sales_date'])
        # aggregate monthly
        sales_df_barea_region = sales_df_barea_region.resample(rule = 'M', on = 'sales_date').sum().reset_index()
        # set timestamp index
        sales_df_barea_region.index = sales_df_barea_region['sales_date']
        
        return sales_df_barea_region
    
    def add_monthly_financial_info(self,
                                   sales_df,
                                   region):
        """
        Add exogenous features to the sales time series data
        """
        
        date_range = pd.date_range(sales_df.index[0], sales_df.index[-1], freq = 'M')
        
        # fill missing values for financial indexes
        fin_idx_df = self.financial_index_df.reindex(date_range)
        # either interpolate or ffill
        fin_idx_df = fin_idx_df.interpolate(method = 'linear')
        fin_idx_df.bfill(inplace = True)
        
        # fill missing values for mgdp
        gdp_df = self.mgdp_df[self.mgdp_df['Sales Area'] == region]
        gdp_df = gdp_df.reindex(date_range)
        # either interpolate or ffill
        gdp_df = gdp_df.interpolate(method = 'linear')
        gdp_df.bfill(inplace = True)
        
        # merge mgdp and financial indexes with sales
        gdp_fin_idx_merged = pd.merge(gdp_df, fin_idx_df,
                                      how = 'inner', left_index = True, right_index = True)
        sales_gdp_fin_merged = pd.merge(gdp_fin_idx_merged, sales_df,
                                        how = 'inner', left_index = True, right_index = True)
        
        # convert timestamp from end of month, to the first day of the next month for joining with commodities and feets
        sales_gdp_fin_merged.index = sales_gdp_fin_merged.index + pd.Timedelta(days = 1)
        
        # pivot commodities table
        comodity_df_pivot = self.comodity_prices_df.pivot_table(index = self.comodity_prices_df.index, columns = ['commodity_name'], values = ['value'])
        comodity_df_pivot.columns = list(comodity_df_pivot.columns.droplevel(level = 0))
        comodity_df_pivot = comodity_df_pivot.loc['2017-02':]
        # merge commodities with sales, financial idxs, and mgdp
        sales_all_exog_merged = pd.merge(sales_gdp_fin_merged,
                                         comodity_df_pivot,
                                         how = 'inner', left_index = True, right_index = True)
        
        # select the fleets for the current region
        fleet_agg = self.fleets_df[self.fleets_df['SalesArea'] == region]
        # aggregate monthly
        fleet_agg = fleet_agg.resample(rule = 'M', on = 'date').sum()
        fleet_agg.index = fleet_agg.index + pd.Timedelta(days = 1)
        # merge with fleets
        sales_all_exog_merged = pd.merge(sales_all_exog_merged,
                                         fleet_agg[['fleet1', 'fleet2', 'fleet3']],
                                         how = 'inner', left_index = True, right_index = True)
        
        return sales_all_exog_merged
