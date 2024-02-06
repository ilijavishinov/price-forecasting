from utils import correlations
import seaborn as sns
import sweetviz
import pandas as pd
import os

from matplotlib import pyplot as plt

PLOTS_DIR = 'plots'
PRICING_COLS = [
    'Average customer net price',
    'List price',
    'Standard Cost',
    'Quantity sold',
]


def ensure_dir_exists(dir_path,
                      check_parent_dir = False):
    """
    Creates a directory if it does not exist
    Can be made recursve if needed
    """
    os.makedirs(dir_path, exist_ok = True)


def get_x_y_relevant_dfs(product,
                         y_name = 'Standard Cost'):
    
    df = pd.read_excel('../data/Price Management.xlsx')
    
    match product:
        case 'drill':
            df = df[df['Product Line Name'] == 'Surface top hammer drill rigs'].iloc[:-2].reset_index(drop = True)
            X_columns = ['Hole diameter min, mm',
                         'Hole diameter max, mm',
                         'Flushing air, m3/min',
                         'Weight, kg',
                         'Drilling coverage, m2',
                         'Percussion power std, kW',]
        case 'loader':
            df = df[df['Product Line Name'] == 'Surface top hammer drill rigs'].iloc[:-2].reset_index(drop = True)
            X_columns = ['Bucket volume std, m3',
                         'Tramming Capacity, t',
                         'Weight Operating, t',
                         'Tilt force, t',
                         'Engine power, kW',
                         'Lift force, t']
        case _:
            raise NameError(rf'Product {product} does not exist! Supported names are [drill, loader]')
    
    X = df[X_columns]
    y = df[y_name]
    relevant = df[X_columns + PRICING_COLS]
    return X, y, relevant
    

def sweetviz_analysis(df,
                      product):
    
    report_df_ckd = sweetviz.analyze(df)
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'general'))
    report_df_ckd.show_html(os.path.join(PLOTS_DIR, 'general', f'sweetviz_{product}.html'))
    

def pairplot(df,
             product):
    
    sns.pairplot(df,
                 kind = 'reg',
                 plot_kws = {'scatter_kws': {'s': 20}})
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'general'))
    plt.savefig(os.path.join(PLOTS_DIR, 'general', f'pairplot_{product}.pdf'))
    plt.close()
    
    
def corrmatrix(df,
               product):

    corr = df.corr()
    sns.heatmap(corr,
                vmin=-1.0, vmax=1.0,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'general'))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'general', f'corr_{product}.pdf'))
    plt.close()


def corrmatrix_scaled(df,
                      product):

    correlation_matrix = df.corr(method = 'pearson')
    p_values_matrix = df.corr(method = correlations.p_value_pearson_nan_proof)
    
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'general'))
    correlations.heatmap_scaled_by_pvalue([correlation_matrix, p_values_matrix],
                                          save_path = os.path.join(PLOTS_DIR, 'general', f'corr_scaled_{product}.pdf'))


