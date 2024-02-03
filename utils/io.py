import os
import pandas as pd


class DataHandler(object):
    """
    """
    
    def __init__(self,
                 data_path):
        self.data_path = data_path
        
        self.comodity_prices = pd.read_csv(os.path.join(data_path, 'commodity_long.csv'))
        
        self.financial_index = pd.read_csv(os.path.join(data_path, 'comp fin index.csv'))
        self.financial_index.index = pd.DatetimeIndex(self.financial_index.date)
        self.financial_index.drop(['date', 'Unnamed: 0'], axis = 1, inplace = True)
        
        self.fleet = pd.read_csv(os.path.join(data_path, 'fleet feats test.csv'))
        
        self.gdp = pd.read_csv(os.path.join(data_path, 'mgdp index.csv'))
        self.gdp.index = pd.DatetimeIndex(self.gdp.date)
        self.gdp.drop(['date', 'Unnamed: 0'], axis = 1, inplace = True)
        
        self.df_sales = pd.read_csv(os.path.join(data_path, 'test sales.csv'))
        self.df_sales['sales_date'] = pd.to_datetime(self.df_sales['sales_date'])
        
    def get_bussines_areas(self):
        """
        """
        return list(set(self.df_sales['business_area_id'].unique().tolist()) - {'Other'})
    
    def get_regions(self):
        """
        """
        return self.df_sales['region_id'].unique().tolist()
        
    
    def get_sales_per_business_area_and_region(self,
                                               business_area,
                                               region):
        """
        """
        sales_bus_region = self.df_sales[(self.df_sales['region_id'] == region) & (self.df_sales['business_area_id'] == business_area)].sort_values(['sales_date'])
        sales_bus_region = sales_bus_region.resample(rule='M', on='sales_date').sum().reset_index()
        sales_bus_region.index = sales_bus_region['sales_date']
        return sales_bus_region
        
    def add_monthly_financial_info(self,
                                   sales_df,
                                   region):
        """
        """
        
        idx = pd.date_range(sales_df.index[0], sales_df.index[-1], freq = 'M')
        
        # fin_idx_df = self.financial_index.reindex(idx, method = 'ffill').reset_index()
        fin_idx_df = self.financial_index.reindex(idx)
        fin_idx_df = fin_idx_df.interpolate(method = 'linear')
        fin_idx_df.bfill(inplace = True)
        
        # fin_idx_df.index = fin_idx_df['index']
        # fin_idx_df.drop(['index'], axis = 1, inplace = True)
        
        gdp_df = self.gdp[self.gdp['Sales Area'] == region]
        # gdp_df = gdp_df.reindex(idx, method = 'ffill').reset_index()
        gdp_df = gdp_df.reindex(idx)
        gdp_df = gdp_df.interpolate(method = 'linear')
        gdp_df.bfill(inplace = True)
        # gdp_df.index = gdp_df['index']
        # gdp_df.drop(['index'], axis = 1, inplace = True)
        
        gdp_fin_idx_merged = pd.merge(gdp_df, fin_idx_df, how = 'inner', left_index = True, right_index = True)
        sales_all_fin_merged = pd.merge(gdp_fin_idx_merged, sales_df, how = 'inner', left_index = True, right_index = True)
        
        # sales_all_fin_merged.index = sales_all_fin_merged.index + pd.Timedelta(days = 1)
        
        return sales_all_fin_merged
    
# def read_ts_files():
#     """
#     """
#
#     comodity_prices = pd.read_csv('data/commodity_long.csv')
#     financial_index = pd.read_csv('data/comp fin index.csv')
#     fleet = pd.read_csv('data/fleet feats test.csv')
#     gdp = pd.read_csv('data/mgdp index.csv')
#     df_sales = pd.read_csv('data/test sales.csv')
#
#     return comodity_prices, financial_index, fleet, gdp, df_sales









