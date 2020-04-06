from .base import *
from .vector import *

from statsmodels.stats.proportion import proportion_confint
from numpy import sqrt, power

# ------------------------------------------------------------------------------
# Series summary functions
# ------------------------------------------------------------------------------


@make_symbolic
def mean(series):
    """
    Returns the mean of a series.

    Args:
        series (pandas.Series): column to summarize.
    """

    # if np.issubdtype(series.dtype, np.number):
    #     return series.mean()
    # else:
    #     return np.nan
    return series.mean()


@make_symbolic
def first(series, order_by=None):
    """
    Returns the first value of a series.

    Args:
        series (pandas.Series): column to summarize.

    Kwargs:
        order_by: a pandas.Series or list of series (can be symbolic) to order
            the input series by before summarization.
    """

    if order_by is not None:
        series = order_series_by(series, order_by)
    first_s = series.iloc[0]
    return first_s


@make_symbolic
def last(series, order_by=None):
    """
    Returns the last value of a series.

    Args:
        series (pandas.Series): column to summarize.

    Kwargs:
        order_by: a pandas.Series or list of series (can be symbolic) to order
            the input series by before summarization.
    """

    if order_by is not None:
        series = order_series_by(series, order_by)
    last_s = series.iloc[series.size - 1]
    return last_s


@make_symbolic
def nth(series, n, order_by=None):
    """
    Returns the nth value of a series.

    Args:
        series (pandas.Series): column to summarize.
        n (integer): position of desired value. Returns `NaN` if out of range.

    Kwargs:
        order_by: a pandas.Series or list of series (can be symbolic) to order
            the input series by before summarization.
    """

    if order_by is not None:
        series = order_series_by(series, order_by)
    try:
        return series.iloc[n]
    except:
        return np.nan


@make_symbolic
def n(series):
    """
    Returns the length of a series.

    Args:
        series (pandas.Series): column to summarize.
    """

    n_s = series.size
    return n_s


@make_symbolic
def n_distinct(series):
    """
    Returns the number of distinct values in a series.

    Args:
        series (pandas.Series): column to summarize.
    """

    n_distinct_s = series.unique().size
    return n_distinct_s


@make_symbolic
def IQR(series):
    """
    Returns the inter-quartile range (IQR) of a series.

    The IRQ is defined as the 75th quantile minus the 25th quantile values.

    Args:
        series (pandas.Series): column to summarize.
    """

    iqr_s = series.quantile(0.75) - series.quantile(0.25)
    return iqr_s


@make_symbolic
def quant(series, p=None):
    """
    Returns the specified quantile value.

    @param series column to summarize
    @param p quantile

    @type series Pandas series
    @type p float

    @pre 0 <= p <= 1
    """

    return series.quantile(p)


@make_symbolic
def colmin(series):
    """
    Returns the minimum value of a series.

    Args:
        series (pandas.Series): column to summarize.
    """

    min_s = series.min()
    return min_s


@make_symbolic
def colmax(series):
    """
    Returns the maximum value of a series.

    Args:
        series (pandas.Series): column to summarize.
    """

    max_s = series.max()
    return max_s


@make_symbolic
def median(series):
    """
    Returns the median value of a series.

    Args:
        series (pandas.Series): column to summarize.
    """

    # if np.issubdtype(series.dtype, np.number):
    #     return series.median()
    # else:
    #     return np.nan
    return series.median()


@make_symbolic
def var(series):
    """
    Returns the variance of values in a series.

    Args:
        series (pandas.Series): column to summarize.
    """
    # if np.issubdtype(series.dtype, np.number):
    #     return series.var()
    # else:
    #     return np.nan
    return series.var()


@make_symbolic
def sd(series):
    """
    Returns the standard deviation of values in a series.

    Args:
        series (pandas.Series): column to summarize.
    """

    # if np.issubdtype(series.dtype, np.number):
    #     return series.std()
    # else:
    #     return np.nan
    return series.std()


@make_symbolic
def colsum(series):
    """
    Returns the sum of values in a series.

    Args:
        series (pandas.Series): column to summarize.
    """

    return series.sum()


@make_symbolic
def mse(series_pred, series_meas):
    """Compute MSE

    Returns the mean-square-error (MSE) between predicted and measured
    values.

    Args:
        series_pred (pandas.Series): column of predictions
        series_meas (pandas.Series): column of predictions

    """

    return (series_pred - series_meas).pow(2).mean()


@make_symbolic
def rmse(series_pred, series_meas):
    """Compute RMSE

    Returns the root-mean-square-error (RMSE) between predicted and measured
    values.

    Args:
        series_pred (pandas.Series): column of predictions
        series_meas (pandas.Series): column of predictions

    """

    return sqrt((series_pred - series_meas).pow(2).mean())


@make_symbolic
def rel_mse(series_pred, series_meas):
    """Compute MSE

    Returns the relative mean-square-error (MSE) between predicted and measured
    values.

    Args:
        series_pred (pandas.Series): column of predictions
        series_meas (pandas.Series): column of predictions

    """

    return ((series_pred - series_meas) / series_meas).pow(2).mean()


@make_symbolic
def rsq(series_pred, series_meas):
    """Compute coefficient of determination

    Returns the coefficient of determination (aka R^2) between predicted and
    measured values.

    Args:
        series_pred (pandas.Series): column of predictions
        series_meas (pandas.Series): column of predictions

    """

    return (
        power(series_pred - series_meas.mean(), 2).sum()
        / power(series_meas - series_meas.mean(), 2).sum()
    )


@make_symbolic
def ndme(series_pred, series_meas):
    """Compute non-dimensional model error

    Returns the non-dimensional model error (NDME) between predicted and
    measured values.

    Args:
        series_pred (pandas.Series): column of predictions
        series_meas (pandas.Series): column of predictions

    """

    return sqrt(1 - rsq(series_pred, series_meas))


@make_symbolic
def binomial_ci(series, alpha=0.05, method="wilson", side="both"):
    """Returns a binomial confidence interval

    Computes a binomial confidence interval based on boolean data. A symbolic
    wrapper for statsmodels.stats.proportion.proportion_confint.

    Args:
        series (pandas.Series): Column to summarize; must be boolean or 0/1.
        alpha (float): Confidence level; value in (0, 1)
        method (string): Method for computation. Options:
            - "normal": asymptotic normal approximation
            - "agresti_coull": Agresti-Coull interval
            - "beta": Clopper-Pearson interval based on Beta distribution
            - "wilson": Wilson score interval
            - "jeffreys": Jeffreys Bayesian interval
        side (string): Chosen side of interval
            - "both": Return a series of tuples
            - "lo": Return the lower interval bound
            - "up": Return the upper interval bound
    """
    count = series.sum()
    nobs = len(series)

    lo, up = proportion_confint(count, nobs, alpha=alpha, method=method)

    if side == "both":
        return (lo, up)
    elif side == "lo":
        return lo
    elif side == "up":
        return up
    else:
        raise ValueError("side value {} not recognized".format(side))
