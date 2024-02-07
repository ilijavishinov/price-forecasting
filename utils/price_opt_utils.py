import itertools
from matplotlib import pyplot as plt
import seaborn as sns
import sweetviz
import pandas as pd
import os
from utils.correlations import *
from utils.metrics import *
from utils.io import *

PLOTS_DIR = 'plots'
PRICING_COLS = [
    'Average customer net price',
    'List price',
    'Standard Cost',
    'Quantity sold',
]


def get_x_y_relevant_dfs(product,
                         y_name = 'Standard Cost'):
    """
    """
    
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
            df = df[df['Product Line Name'] != 'Surface top hammer drill rigs'].reset_index(drop = True)
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
    """
    """
    
    report_df_ckd = sweetviz.analyze(df)
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'general'))
    report_df_ckd.show_html(os.path.join(PLOTS_DIR, 'general', f'sweetviz_{product}.html'))
    

def pairplot(df,
             product):
    """
    """
    
    sns.pairplot(df,
                 kind = 'reg',
                 plot_kws = {'scatter_kws': {'s': 20}})
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'general'))
    plt.savefig(os.path.join(PLOTS_DIR, 'general', f'pairplot_{product}.pdf'))
    plt.close()
    
    
def corrmatrix(df,
               product):
    """
    """

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
    """
    """

    correlation_matrix = df.corr(method = 'pearson')
    p_values_matrix = df.corr(method = p_value_pearson_nan_proof)
    
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'general'))
    heatmap_scaled_by_pvalue([correlation_matrix, p_values_matrix],
                             save_path = os.path.join(PLOTS_DIR, 'general', f'corr_scaled_{product}.pdf'))


def linear_regression_coefficients(model, X):
    coefs = {X.columns[i]: model.coef_[i].round(4) for i in range(X.shape[1])}
    print('')
    return coefs
    

def linear_regression_formula(model, X, y_name = ''):
    coefs = {X.columns[i]: model.coef_[i].round(4) for i in range(X.shape[1])}
    intercept = model.intercept_
    formula_str = f'{y_name} = {int(intercept)} '
    for coef_name, coef_value in coefs.items():
        formula_str += f'{"+" if coef_value >= 0 else "-"} {int(abs(coef_value))} {coef_name} '
    formula_str = formula_str[:-3]
    return formula_str


def iterative_selectiono_vif_ordered_features(x):
    """
    """
    x = x.copy(deep = True)
    
    iterative_selected_features = [x.columns.tolist()]
    num_features = x.shape[1]
    while len(iterative_selected_features) < num_features:
        vif_df = calc_vif(x)
        x.drop(vif_df.max()['feature'], axis = 1, inplace = True)
        iterative_selected_features.append(x.columns.tolist())
    
    return iterative_selected_features


def all_possible_feature_combinations(x):
    """
    """
    features = x.columns.tolist()

    all_combinations_of_all_lengths = list()
    for length in range(len(features) + 1):
        for subset in itertools.combinations(features, length):
            if len(subset) > 0:
                all_combinations_of_all_lengths.append(list(subset))
            
    return all_combinations_of_all_lengths



