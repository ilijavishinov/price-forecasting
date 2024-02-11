import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np


def p_value_pearson_nan_proof(x, y, round_flag: bool = False):
    
    two_feature_frame = pd.DataFrame({'x': x, 'y': y})
    where_both_not_null_frame = two_feature_frame[~two_feature_frame['x'].isna() & ~two_feature_frame['y'].isna()]
    if where_both_not_null_frame.shape[0] < 3:
        return np.nan
    else:
        return_val =  stats.pearsonr(where_both_not_null_frame['x'], where_both_not_null_frame['y'])[1]
        if round_flag: return round(return_val, 4)
        else: return return_val


def series_apply_pval(s):
    return s.apply(p_values_apply)


def p_values_apply(p_val):
    if p_val <= 0.05: return 1
    elif p_val <= 0.1: return .25
    else: return .05


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
                             ) -> None:
    
    """
    Plot a correlation matrix with correlation cells scaled by p-value
    """
    
    # defaults
    plt.rcParams["figure.figsize"] = figsize
    
    if not annot_white_treshold:
        annot_white_treshold = max_corr * 0.7
    
    # # define plot structure
    fig, ax = plt.subplots()
    
    # take correlation and p value matrix
    corr = corr_pval_list[0]
    # corr = corr[corr.columns.tolist()[::-1]]
    pval = corr_pval_list[1]
    # pval = pval[pval.columns.tolist()[::-1]]
    
    # scaling cells by p-val
    pval = pval.apply(series_apply_pval)
    
    # melt the dataframe, so scatter plot can be drawn
    corr = pd.melt(corr.reset_index(), id_vars = 'index')
    corr.columns = ['x', 'y', 'value']
    pval = pd.melt(pval.reset_index(), id_vars = 'index')
    pval.columns = ['x', 'y', 'p-value']
    
    # x goes on x axis, y goes on y axis
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
        
        if size.at[index_annot] == 1:
            # temps
            annot_fontsize_temp = annot_fontsize
            
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

