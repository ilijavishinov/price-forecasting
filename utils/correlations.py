import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np


def pearson_nan_proof(x, y, round_flag: bool = False):
    
    two_feature_frame = pd.DataFrame({'x': x, 'y': y})
    where_both_not_null_frame = two_feature_frame[~two_feature_frame['x'].isna() & ~two_feature_frame['y'].isna()]
    if where_both_not_null_frame.shape[0] < 3:
        return np.nan
    else:
        return_val =  stats.pearsonr(where_both_not_null_frame['x'], where_both_not_null_frame['y'])[0]
        if round_flag: return round(return_val, 4)
        else: return return_val


def spearman_nan_proof(x, y, round_flag: bool = False):
    
    two_feature_frame = pd.DataFrame({'x': x, 'y': y})
    where_both_not_null_frame = two_feature_frame[~two_feature_frame['x'].isna() & ~two_feature_frame['y'].isna()]
    if where_both_not_null_frame.shape[0] < 3:
        return np.nan
    else:
        return_val = stats.spearmanr(where_both_not_null_frame['x'], where_both_not_null_frame['y'])[0]
        if round_flag: return round(return_val, 4)
        else: return return_val


def p_value_pearson_nan_proof(x, y, round_flag: bool = False):
    
    two_feature_frame = pd.DataFrame({'x': x, 'y': y})
    where_both_not_null_frame = two_feature_frame[~two_feature_frame['x'].isna() & ~two_feature_frame['y'].isna()]
    if where_both_not_null_frame.shape[0] < 3:
        return np.nan
    else:
        return_val =  stats.pearsonr(where_both_not_null_frame['x'], where_both_not_null_frame['y'])[1]
        if round_flag: return round(return_val, 4)
        else: return return_val


def p_value_spearman_nan_proof(x, y, round_flag: bool = False):
    
    two_feature_frame = pd.DataFrame({'x': x, 'y': y})
    where_both_not_null_frame = two_feature_frame[~two_feature_frame['x'].isna() & ~two_feature_frame['y'].isna()]
    if where_both_not_null_frame.shape[0] < 3:
        return np.nan
    else:
        return_val =  stats.spearmanr(where_both_not_null_frame['x'], where_both_not_null_frame['y'])[1]
        if round_flag: return round(return_val, 4)
        else: return return_val



def target_correlations(df: pd.DataFrame, _method: str, ):
    # choose target feature
    if _method == 'PBC': target_feature = 'Class'
    else: target_feature = 'Glucose'
    
    # different correlation methods
    if _method == 'Spearman':
        cmtx = df.corr(method = 'spearman')
        pmtx = df.corr(method = p_value_spearman_nan_proof)
    else:
        cmtx = df.corr(method = 'pearson')
        pmtx = df.corr(method = p_value_pearson_nan_proof)
    
    return cmtx[target_feature], pmtx[target_feature]

def series_apply_pval(s):
    return s.apply(p_values_apply)

def series_apply_percents(s):
    return s.apply(percents_apply)

def p_values_apply(p_val):
    if p_val <= 0.05: return 1
    elif p_val <= 0.1: return .25
    else: return .05


def percents_apply(percent):
    if abs(percent) <= 0.01: return .05
    if abs(percent) <= 0.05: return .25
    else: return 1
    
    # nested function for mapping the correlation value to a color

def value_to_color(val):
    
    color_max = 1
    color_min = -1
    n_colors = 256
    palette = sns.color_palette("RdBu_r", n_colors = n_colors)

    # if outside of correlation scale
    if val >= color_max:
        return palette[-1]
    if val <= color_min:
        return palette[0]
    
    # position of value in the input range, relative to the length of the input range
    val_position = float((val - color_min)) / (color_max - color_min)
    # target index in the color palette
    ind = int(val_position * (n_colors - 1))
    
    return palette[ind]

def heatmap_scaled_by_pvalue(corr_pval_list,
                             save_path: str = None,
                             size_scale: int = 500,
                             max_corr: float = 1,
                             annot_shift_x: float = .12,
                             annot_shift_y: float = .12,
                             annot_white_treshold: float = None,
                             annot_fontsize: int = 15,
                             xtickslabels_fontsize: int = 15,
                             yticklabels_fontsize: int = 15,
                             figsize = (15, 8),
                             show_point: bool = True,
                             percentages: bool = False,
                             ) -> None:
    
    """
    Plot a correlation matrix with correlation cells scaled by p-value
    """
    
    # defaults
    plt.rcParams["figure.figsize"] = figsize
    if percentages: show_point = False
    
    if not annot_white_treshold:
        annot_white_treshold = max_corr * 0.7
    
    # # define plot structure
    fig, ax = plt.subplots()
    
    # take correlation and p value matrix
    corr = corr_pval_list[0]
    # corr = corr[corr.columns.tolist()[::-1]]
    pval = corr_pval_list[1]
    # pval = pval[pval.columns.tolist()[::-1]]
    
    # scaling cells by percentage
    if percentages:
        # consider p values and improvement percentage scaling
        pval_scaling = pval.apply(series_apply_pval)
        percent_scaling = corr.apply(series_apply_percents)
        
        # scale by the minimum of the two values
        pval = np.minimum(pval_scaling, percent_scaling)
    # scaling cells by p-val
    else:
        pval = pval.apply(series_apply_pval)
    
    # melt the dataframe, so scatter plot can be drawn
    corr = pd.melt(corr.reset_index(), id_vars = 'index')
    corr.columns = ['x', 'y', 'value']
    pval = pd.melt(pval.reset_index(), id_vars = 'index')
    pval.columns = ['x', 'y', 'p-value']
    
    # x goes on x axis
    # y goes on y axis
    # size is determined by the p-value
    x = corr['x']
    y = corr['y']
    size = pval['p-value']
    
    # define labels and positions on scatter axis
    x_labels = [v for v in x.unique()]
    y_labels = [v for v in y.unique()]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
    
    # limits of the plot
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    
    # take nans as zero correlations
    corr.replace(to_replace = {np.NaN: 0}, inplace = True)
    
    # add color column
    # arg_c = value_to_color_corr(corr)
    corr['color'] = corr['value'].apply(value_to_color)
    
    ax.scatter(x = x.map(x_to_num),
               y = y.map(y_to_num),
               s = size * size_scale,
               c = corr['color'],
               marker = 's')
    
    # put correlation annotations in each cell
    for index_annot in range(corr.shape[0]):
        
        # default
        color = 'black'
        
        if size.at[index_annot] == 1:
            # temps
            annot_fontsize_temp = annot_fontsize
            reduce_y_shift_by = 0
            
            # used for displaying percentages
            if not show_point:
                if corr['value'].at[index_annot] >= 1:
                    annotation = str(round(corr['value'].at[index_annot], 2)).replace('.', '')
                    if len(annotation) == 2: annotation += '0'
                    annot_fontsize_temp = annot_fontsize - 4
                    if corr['value'].at[index_annot] >= annot_white_treshold:
                        color = 'white'
                    else:
                        color = 'black'
                
                elif corr['value'].at[index_annot] > 0:
                    annotation = str(round(corr['value'].at[index_annot], 2))[2:]
                    if corr['value'].at[index_annot] >= annot_white_treshold:
                        color = 'white'
                    else:
                        color = 'black'
                
                elif corr['value'].at[index_annot] == 0:
                    annotation = '00'
                
                else:
                    annotation = str(round(corr['value'].at[index_annot], 2))[3:]
                    if corr['value'].at[index_annot] <= -annot_white_treshold:
                        color = 'white'
                    else:
                        color = 'black'
                
                if len(annotation) == 1:
                    annotation += '0'
            
            # for positive floats
            else:
                
                if corr['value'].at[index_annot] >= 0:
                    
                    annotation = str(round(corr['value'].at[index_annot], 2))[1:]
                    if corr['value'].at[index_annot] >= annot_white_treshold:
                        color = 'white'
                    else:
                        color = 'black'
                
                # negative floats
                else:
                    
                    annotation = str(round(corr['value'].at[index_annot], 2))[2:]
                    if corr['value'].at[index_annot] <= -annot_white_treshold:
                        color = 'w'
                    else:
                        color = 'black'
                
                if len(annotation) == 2:
                    annotation += '0'
            
            # annotate the point on the scatter plot
            ax.annotate(text = annotation,
                        fontstyle = 'normal', fontsize = annot_fontsize_temp,
                        fontweight = 'ultralight', c = color,
                        xy = (x.map(x_to_num).at[index_annot] - annot_shift_x,
                              y.map(y_to_num).at[index_annot] - annot_shift_y))
    
    # Show column labels on the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation = 55, ha = 'right', fontsize = xtickslabels_fontsize)
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels([v.replace(' min window', '') for v in y_labels], fontsize = yticklabels_fontsize)
    
    plt.subplots_adjust(top = .93, bottom = .07, left = .06, right = .94, hspace = .3, wspace = .25)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path.replace('SD1/SD2', 'SD1divSD2').replace('(‰)', ''))
    else:
        plt.show()
    
    plt.clf()
    plt.close('all')

#____________________________________________________________________________________________________________________________________


def bar_plot_vertical_short(corr_limit: float,
                            plot_save_path: str = None
                            ) -> None:
    """
    Generate a heatmap barplot showing the scale of the correlations

    :param corr_limit: abs maximum for the correlations
    :param plot_save_path: path for saving the figure. If None, the plot is shown.
    """
    
    fig, ax = plt.subplots()
    
    n_colors = 256
    palette = sns.color_palette("RdBu_r", n_colors=n_colors)
    max_corr_val_group = corr_limit
    
    if max_corr_val_group is not None: color_max, color_min = [max_corr_val_group * x for x in [1, -1]]
    else: color_max, color_min = [var for var in [1, -1]]
    
    col_x = [0]*len(palette)
    
    bar_y = np.linspace(color_min, color_max, n_colors)
    
    bar_height = bar_y[1] - bar_y[0]
    
    ax.barh(
        y=bar_y,
        width=2,
        left=col_x,
        height=bar_height,
        color=palette,
        linewidth=0,
        align = 'edge'
    )
    
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    ax.set_xlim(1, 2)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
    ax.yaxis.tick_right()
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10000)
    
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.06, right=0.2, hspace=0.25, wspace=2.5)
    plt.tight_layout()
    
    if plot_save_path is None: plt.show()
    else: plt.savefig(plot_save_path)
    
    plt.show()
    plt.clf()
    plt.close('all')


