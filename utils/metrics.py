from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def regression_metrics(y_true, y_pred):
    """
    """
    return dict(mae = mean_absolute_error(y_true, y_pred),
                mape = mean_absolute_percentage_error(y_true, y_pred),
                rmse = root_mean_squared_error(y_true, y_pred))


def calc_vif(df):
    """
    Measure for the increase of the variance of the parameter estimates if an additional variable is added to the linear regression
    It is a measure for multicollinearity of the design matrix

    One recommendation is that if VIF is greater than 5
    - Then the explanatory variable given by exog_idx is highly collinear with the other explanatory variables
        - The parameter estimates will have large standard errors because of this

    """
    # Calculating VIF
    vif = pd.DataFrame()
    vif["feature"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif






