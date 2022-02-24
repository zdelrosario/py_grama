__all__ = [
    "binomial_ci",
    "corr",
    "mean",
    "mean_lo",
    "mean_up",
    "IQR",
    "quant",
    "pr",
    "pr_lo",
    "pr_up",
    "var",
    "sd",
    "skew",
    "kurt",

    "min",
    "max",
    "sum",
    "median",

    "first",
    "last",

    "n",
    "nth",
    "n_distinct",
    "neff_is",

    "mse",
    "rmse",
    "ndme",
    "rsq",
]

from .base import make_symbolic
from .vector import order_series_by
from numpy import array, sqrt, power, nan, isnan
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


# Mean CI helpers
@make_symbolic
def mean_lo(series, alpha=0.01):
    """Return a confidence interval (lower bound) for the mean

    Uses a central limit approximation for a lower confidence bound of an estimated mean. That is:

        m - q(alpha) * s / sqrt(n)

    where

        m = sample mean
        s = sample standard deviation
        n = sample size
        q(alpha) = alpha-level lower-quantile of standard normal
                 = (-norm.ppf(alpha))

    Args:
        series (pandas.Series): column to summarize
        alpha (float): alpha-level for calculation
            Note that the confidence level C is given by C = 1 - alpha

    Returns:
        float: Lower confidence interval for the mean
    """
    m = mean(series)
    s = sd(series)
    n_sample = len(series)

    return m - (-norm.ppf(alpha)) * s / sqrt(n_sample)


@make_symbolic
def mean_up(series, alpha=0.01):
    """Return a confidence interval (upper bound) for the mean

    Uses a central limit approximation for a upper confidence bound of an estimated mean. That is:

        m + q(alpha) * s / sqrt(n)

    where

        m = sample mean
        s = sample standard deviation
        n = sample size
        q(alpha) = alpha-level lower-quantile of standard normal
                 = (-norm.ppf(alpha))

    Args:
        series (pandas.Series): column to summarize
        alpha (float): alpha-level for calculation
            Note that the confidence level C is given by C = 1 - alpha

    Returns:
        float: Upper confidence interval for the mean
    """
    m = mean(series)
    s = sd(series)
    n_sample = len(series)

    return m + (-norm.ppf(alpha)) * s / sqrt(n_sample)


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
def min(series):
    """
    Returns the minimum value of a series.

    Args:
        series (pandas.Series): column to summarize.
    """

    min_s = series.min()
    return min_s


@make_symbolic
def max(series):
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
def sum(series):
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

# Probability helpers
# --------------------------------------------------
@make_symbolic
def pr(series):
    """Estimate a probability

    Estimate a probability from a random sample. Provided series must be boolean, with 1 corresponding to the event of interest.

    Use logical statements together with column values to construct a boolean indicator for the event you're interested in. Remember that you can chain multiple statements with logical and `&` and or `|` operators. See the examples below for more details.

    Args:
        series (pandas.Series): Column to summarize; must be boolean or 0/1.

    Examples:
        >>> import grama as gr
        >>> DF = gr.Intention()
        >>> ## Cantilever beam examples
        >>> from grama.models import make_cantilever_beam
        >>> md_beam = make_cantilever_beam()
        >>>
        >>> ## Estimate probabilities
        >>> (
        >>>     md_beam
        >>>     # Generate large
        >>>     >> gr.ev_sample(n=1e5, df_det="nom")
        >>>     # Estimate probabilities of failure
        >>>     >> gr.tf_summarize(
        >>>         pof_stress=gr.pr(DF.g_stress <= 0),
        >>>         pof_disp=gr.pr(DF.g_disp <= 0),
        >>>         pof_joint=gr.pr( (DF.g_stress <= 0) & (DF.g_disp) ),
        >>>         pof_either=gr.pr( (DF.g_stress <= 0) | (DF.g_disp) ),
        >>>     )
        >>> )

    """
    return series.mean()


@make_symbolic
def pr_lo(series, alpha=0.01):
    r"""Estimate a confidence interval for a probability

    Estimate the lower side of a confidence interval for a probability from a random sample. Provided series must be boolean, with 1 corresponding to the event of interest.

    Uses Wilson interval method.

    Use logical statements together with column values to construct a boolean indicator for the event you're interested in. Remember that you can chain multiple statements with logical and `&` and or `|` operators. See the documentation for `gr.pr()` for more details and examples.

    Args:
        series (pandas.Series): Column to summarize; must be boolean or 0/1.
        alpha (float): alpha-level for calculation, in (0, 1)
            Note that the confidence level C is given by C = 1 - alpha

    Returns:
        float: Lower confidence interval

    Examples:
        >>> import grama as gr
        >>> DF = gr.Intention()
        >>> ## Cantilever beam examples
        >>> from grama.models import make_cantilever_beam
        >>> md_beam = make_cantilever_beam()
        >>>
        >>> ## Estimate probabilities
        >>> (
        >>>     md_beam
        >>>     # Generate large
        >>>     >> gr.ev_sample(n=1e5, df_det="nom")
        >>>     # Estimate probabilities with a confidence interval
        >>>     >> gr.tf_summarize(
        >>>         pof_lo=gr.pr_lo(DF.g_stress <= 0),
        >>>         pof=gr.pr(DF.g_stress <= 0),
        >>>         pof_up=gr.pr_up(DF.g_stress <= 0),
        >>>     )
        >>> )
    """
    up = binomial_ci(series, alpha=alpha, side="lo")
    return up


@make_symbolic
def pr_up(series, alpha=0.01):
    r"""

    Estimate the upper side of a confidence interval for a probability from a random sample. Provided series must be boolean, with 1 corresponding to the event of interest.

    Uses Wilson interval method.

    Use logical statements together with column values to construct a boolean indicator for the event you're interested in. Remember that you can chain multiple statements with logical and `&` and or `|` operators. See the documentation for `gr.pr()` for more details and examples.

    Args:
        series (pandas.Series): Column to summarize; must be boolean or 0/1.
        alpha (float): alpha-level for calculation, in (0, 1)
            Note that the confidence level C is given by C = 1 - alpha

    Returns:
        float: Upper confidence interval

    Examples:
        >>> import grama as gr
        >>> DF = gr.Intention()
        >>> ## Cantilever beam examples
        >>> from grama.models import make_cantilever_beam
        >>> md_beam = make_cantilever_beam()
        >>>
        >>> ## Estimate probabilities
        >>> (
        >>>     md_beam
        >>>     # Generate large
        >>>     >> gr.ev_sample(n=1e5, df_det="nom")
        >>>     # Estimate probabilities with a confidence interval
        >>>     >> gr.tf_summarize(
        >>>         pof_lo=gr.pr_lo(DF.g_stress <= 0),
        >>>         pof=gr.pr(DF.g_stress <= 0),
        >>>         pof_up=gr.pr_up(DF.g_stress <= 0),
        >>>     )
        >>> )
    """
    up = binomial_ci(series, alpha=alpha, side="up")
    return up


@make_symbolic
def corr(series1, series2, method="pearson", res="corr", nan_drop=False):
    r"""Computes a correlation coefficient

    Computes a correlation coefficient using either the pearson or spearman
    formulation.

    Args:
        series1 (pandas.Series): Column 1 to study
        series2 (pandas.Series): Column 2 to study
        method (str): Method to use; either "pearson" or "spearman"
        res (str): Quantities to return; either "corr" or "both"
        na_drop (bool): Drop NaN values before computation?

    Returns:
        pandas.Series: correlation coefficient

    """
    if nan_drop:
        ids = (isnan(series1) | isnan(series2))
    else:
        ids = array([False] * len(series1))

    if method == "pearson":
        r, p = pearsonr(series1[~ids], series2[~ids])
    elif method == "spearman":
        r, p = spearmanr(series1[~ids], b=series2[~ids])
    else:
        raise ValueError("method {} not supported".format(method))

    if res == "corr":
        return r
    if res == "both":
        return r, p
    else:
        raise ValueError("res {} not supported".format(res))


# ------------------------------------------------------------------------------
# Effective Sample Size helpers
# ------------------------------------------------------------------------------
@make_symbolic
def neff_is(series):
    """Importance sampling n_eff

    Computes the effective sample size according to Equation 9.13 of Owen
    (2013).

    Args:
        series (pandas.Series): column to summarize.

    References:
        A.B. Owen, "Monte Carlo theory, methods and examples" (2013)

    """

    w = series.mean()
    w2 = (series**2).mean()
    return len(series) * w**2 / w2
