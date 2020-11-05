__all__ = [
    "abs",
    "sin",
    "cos",
    "log",
    "exp",
    "sqrt",
    "pow",
    "as_int",
    "as_float",
    "as_str",
    "as_factor",
    "fct_reorder",
    "fillna",
    "qnorm",
    "pnorm",
    "dnorm",
]

from grama import make_symbolic

from numpy import argsort, array, median, zeros
from numpy import abs as npabs
from numpy import sin as npsin
from numpy import cos as npcos
from numpy import log as nplog
from numpy import exp as npexp
from numpy import sqrt as npsqrt
from numpy import power as nppower
from pandas import Categorical, Series
from scipy.stats import norm

# --------------------------------------------------
# Mutation helpers
# --------------------------------------------------
# Numeric
# -------------------------
@make_symbolic
def abs(x):
    r"""Absolute value
    """
    return npabs(x)


@make_symbolic
def sin(x):
    r"""Sine
    """
    return npsin(x)


@make_symbolic
def cos(x):
    r"""Cosine
    """
    return npcos(x)


@make_symbolic
def log(x):
    r"""(Natural) log
    """
    return nplog(x)


@make_symbolic
def exp(x):
    r"""Exponential (e-base)
    """
    return npexp(x)


@make_symbolic
def sqrt(x):
    r"""Square-root
    """
    return npsqrt(x)


@make_symbolic
def pow(x, p):
    r"""Power

    Usage:
        q = pow(x, p) := x ^ p

    Arguments:
        x = base
        p = exponent
    """
    return nppower(x, p)


# Casting
# -------------------------
@make_symbolic
def as_int(x):
    r"""Cast to integer
    """
    return x.astype(int)


@make_symbolic
def as_float(x):
    r"""Cast to float
    """
    return x.astype(float)


@make_symbolic
def as_str(x):
    r"""Cast to string
    """
    return x.astype(str)


@make_symbolic
def as_factor(x, categories=None, ordered=True, dtype=None):
    r"""Cast to factor
    """
    return Categorical(x, categories=categories, ordered=ordered, dtype=dtype)


# Distributions
# -------------------------
@make_symbolic
def qnorm(x):
    r"""Normal quantile function (inverse CDF)
    """
    return norm.ppf(x)


@make_symbolic
def dnorm(x):
    r"""Normal probability density function (PDF)
    """
    return norm.pdf(x)


@make_symbolic
def pnorm(x):
    r"""Normal cumulative distribution function (CDF)
    """
    return norm.cdf(x)


# Factors
# -------------------------
@make_symbolic
def fct_reorder(f, x, fun=median):
    r"""Reorder a factor on another variable

    Usage:
        fp = fct_reorder(f, x)
        fp = fct_reorder(f, x, fun=median)

    Args:
        f (iterable OR DataFrame column): factor to reorder
        x (iterable OR DataFrame column): variable on which to reorder; specify aggregation method with fun
        fun (function): aggregation function for reordering

    Returns:
        Categorical: Iterable with levels sorted according to x

    Examples:
        >>> import grama as gr
        >>> from grama.data import df_diamonds
        >>> X = gr.Intention()
        >>> (
        >>>     df_diamonds
        >>>     >> gr.tf_mutate(cut=gr.fct_reorder(X.cut, X.price, fun=gr.colmax))
        >>>     >> gr.tf_group_by(X.cut)
        >>>     >> gr.tf_summarize(max=gr.colmax(X.price), mean=gr.mean(X.price))
        >>> )
    """

    # Get factor levels
    levels = array(list(set(f)))
    # Compute given fun over associated values
    values = zeros(len(levels))
    for i in range(len(levels)):
        mask = f == levels[i]
        values[i] = fun(x[mask])
    # Sort according to computed values
    return as_factor(f, categories=levels[argsort(values)], ordered=True)


# Pandas helpers
# -------------------------
@make_symbolic
def fillna(*args, **kwargs):
    return Series.fillna(*args, **kwargs)
