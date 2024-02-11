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


def get_x_y_relevant_dfs(product: str,
                         y_name: str = 'Standard Cost'):
    """
    Get X (features), y (target), and relevant df (only with relevant columns) for cleaner analysis
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
    Quick overall analysis with sweetviz
    """
    
    report_df_ckd = sweetviz.analyze(df)
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'general'))
    report_df_ckd.show_html(os.path.join(PLOTS_DIR, 'general', f'sweetviz_{product}.html'))
    

def pairplot(df,
             product):
    """
    Seaborn pairplot to analyze features relationships
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
    Simple correlation matrix
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
    More advanced correlation matrix with squares scaled by p-values
    To distinguish significant from non-significant correlations
    """

    correlation_matrix = df.corr(method = 'pearson')
    p_values_matrix = df.corr(method = p_value_pearson_nan_proof)
    
    ensure_dir_exists(os.path.join(PLOTS_DIR, 'general'))
    heatmap_scaled_by_pvalue([correlation_matrix, p_values_matrix],
                             save_path = os.path.join(PLOTS_DIR, 'general', f'corr_scaled_{product}.pdf'))


def linear_regression_coefficients(model, X_columns):
    """
    Get the coefficients of a linear regression model
    """
    coefs = {X_columns[i]: model.coef_[i].round(4) for i in range(len(X_columns))}
    return coefs
    

def linear_regression_formula(model, X_columns, y_name = ''):
    """
    Get a string formula for the solved linear regression equation
    """
    coefs = {X_columns[i]: model.coef_[i].round(4) for i in range(len(X_columns))}
    intercept = model.intercept_
    formula_str = f'{y_name} = {int(intercept)} '
    for coef_name, coef_value in coefs.items():
        formula_str += f'{"+" if coef_value >= 0 else "-"} {int(abs(coef_value))} {coef_name} '
    formula_str = formula_str[:-3]
    return formula_str


def iterative_selectiono_vif_ordered_features(x):
    """
    Itteratively turn off one feature wich has the highest VIF (is most correlated with other features)
    Goal is to remove multicollinearity for clearer explanation
    """
    x = x.copy(deep = True)
    
    iterative_selected_features = [x.columns.tolist()]
    num_features = x.shape[1]
    while len(iterative_selected_features) < num_features:
        vif_df = calc_vif(x)
        print(vif_df)
        # TODO: wrong selection of vif max feature
        x.drop(vif_df[vif_df['VIF'].idxmax()]['feature'], axis = 1, inplace = True)
        iterative_selected_features.append(x.columns.tolist())
    
    return iterative_selected_features


def all_possible_feature_combinations(x):
    """
    Get all combinations of all lengths from the features
    """
    features = x.columns.tolist()

    all_combinations_of_all_lengths = list()
    for length in range(len(features) + 1):
        for subset in itertools.combinations(features, length):
            if len(subset) > 0:
                all_combinations_of_all_lengths.append(list(subset))
            
    return all_combinations_of_all_lengths

