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
    return npabs(x)


@make_symbolic
def sin(x):
    return npsin(x)


@make_symbolic
def cos(x):
    return npcos(x)


@make_symbolic
def log(x):
    return nplog(x)


@make_symbolic
def exp(x):
    return npexp(x)


@make_symbolic
def sqrt(x):
    return npsqrt(x)


@make_symbolic
def pow(x, p):
    return nppower(x, p)


# Casting
# -------------------------
@make_symbolic
def as_int(x):
    return x.astype(int)


@make_symbolic
def as_float(x):
    return x.astype(float)


@make_symbolic
def as_str(x):
    return x.astype(str)


@make_symbolic
def as_factor(x, categories=None, ordered=True, dtype=None):
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
