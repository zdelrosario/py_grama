__all__ = [
    "mean",
    "first",
    "last",
    "nth",
    "n",
    "n_distinct",
    "IQR",
    "quant",
    "colmin",
    "colmax",
    "colsum",
    "median",
    "var",
    "sd",
    "skew",
    "kurt",
    "binomial_ci",
    "mse",
    "rmse",
    "ndme",
    "rsq",
    "corr",
]

from .base import make_symbolic
from .vector import order_series_by
from numpy import sqrt, power, nan
from scipy.stats import norm, pearsonr, spearmanr, kurtosis
from scipy.stats import skew as spskew


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
def skew(series, bias=True, nan_policy="propagate"):
    """Returns the skewness of a series.

    Args:
        series (pandas.Series): column to summarize.
        bias (bool): Correct for bias?
        nan_policy (str): How to handle NaN values:
            - "propagate": return NaN
            - "raise": throws an error
            - "omit": remove NaN before calculating skew

    """

    return spskew(series, bias=bias, nan_policy="propagate")


@make_symbolic
def kurt(series, bias=True, nan_policy="propagate", excess=False):
    """Returns the kurtosis of a series.

    A distribution with kurtosis greater than three is called *leptokurtic*;
    such a distribution has "fatter" tails and will tend to exhibit more
    outliers. A distribution with kurtosis less than three is called
    *platykurtic*; such a distribution has less-fat tails and will tend to
    exhibit fewer outliers.

    Args:
        series (pandas.Series): column to summarize.
        bias (bool): Correct for bias?
        excess (bool): Return excess kurtosis (excess = kurtosis - 3).
            Note that a normal distribution has kurtosis == 3, which
            informs the excess kurtosis definition.
        nan_policy (str): How to handle NaN values:
            - "propagate": return NaN
            - "raise": throws an error
            - "omit": remove NaN before calculating skew

    """

    return kurtosis(series, fisher=excess, bias=bias, nan_policy="propagate")

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
        return nan


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

    y_mean = series_meas.mean()

    SS_res = power(series_meas - series_pred, 2).sum()
    SS_tot = power(series_meas - y_mean, 2).sum()

    return 1 - SS_res / SS_tot


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
def binomial_ci(series, alpha=0.05, side="both"):
    """Returns a binomial confidence interval

    Computes a binomial confidence interval based on boolean data. Uses Wilson interval

    Args:
        series (pandas.Series): Column to summarize; must be boolean or 0/1.
        alpha (float): Confidence level; value in (0, 1)
        side (string): Chosen side of interval
            - "both": Return a 2-tuple of series
            - "lo": Return the lower interval bound
            - "up": Return the upper interval bound
    """
    n_s = series.sum()
    n_t = len(series)
    n_f = n_t - n_s
    z = -norm.ppf(alpha / 2)

    mid = (n_s + 0.5 * z ** 2) / (n_t + z ** 2)
    delta = z / (n_t + z ** 2) * sqrt(n_s * n_f / n_t + 0.25 * z ** 2)
    lo = mid - delta
    up = mid + delta

    if side == "both":
        return (lo, up)
    if side == "lo":
        return lo
    if side == "up":
        return up
    else:
        raise ValueError("side value {} not recognized".format(side))


@make_symbolic
def corr(series1, series2, method="pearson", res="corr"):
    r"""Computes a correlation coefficient

    Computes a correlation coefficient using either the pearson or spearman
    formulation.

    Args:
        series1 (pandas.Series): Column 1 to study
        series2 (pandas.Series): Column 2 to study
        method (str): Method to use; either "pearson" or "spearman"
        res (str): Quantities to return; either "corr" or "both"

    Returns:
        pandas.Series: correlation coefficient

    """
    if method == "pearson":
        r, p = pearsonr(series1, series2)
    elif method == "spearman":
        r, p = spearmanr(series1, b=series2)
    else:
        raise ValueError("method {} not supported".format(method))

    if res == "corr":
        return r
    if res == "both":
        return r, p
    else:
        raise ValueError("res {} not supported".format(res))
