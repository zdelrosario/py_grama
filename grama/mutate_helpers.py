__all__ = [
    "abs",
    "sin",
    "cos",
    "log",
    "exp",
    "sqrt",
    "pow",
    "floor",
    "ceil",
    "round",
    "as_int",
    "as_float",
    "as_str",
    "as_factor",
    "as_numeric",
    "fct_reorder",
    "fillna",
    "qnorm",
    "pnorm",
    "dnorm",
    "pareto_min",
    "stratum_min",
]

from grama import make_symbolic
from numpy import argsort, array, median, zeros, ones, NaN, arange
from numpy import any as npany
from numpy import all as npall
from numpy import abs as npabs
from numpy import sin as npsin
from numpy import cos as npcos
from numpy import log as nplog
from numpy import exp as npexp
from numpy import sqrt as npsqrt
from numpy import power as nppower
from numpy import floor as npfloor
from numpy import ceil as npceil
from numpy import round as npround
from pandas import Categorical, Series, to_numeric
from scipy.stats import norm


# --------------------------------------------------
# Mutation helpers
# --------------------------------------------------
# Numeric
# -------------------------
@make_symbolic
def floor(x):
    r"""Absolute value
    """
    return npfloor(x)


@make_symbolic
def ceil(x):
    r"""Absolute value
    """
    return npceil(x)


@make_symbolic
def round(x):
    r"""Absolute value
    """
    return npround(x)


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


@make_symbolic
def as_numeric(x):
    r"""Cast to factor
    """
    return to_numeric(x, errors="coerce")


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


# Pareto frontier calculation
# -------------------------
@make_symbolic
def pareto_min(*args):
    r"""Determine if observation is a Pareto point

    Find the Pareto-efficient points that minimize the provided features.

    Args:
        xi (iterable OR gr.Intention()): Feature to minimize; use -X to maximize

    Returns:
        np.array of boolean: Indicates if observation is Pareto-efficient
    """
    # Check invariants
    lengths = map(len, args)
    if len(set(lengths)) > 1:
        raise ValueError("All arguments to pareto_min must be of equal length")

    # Compute pareto points
    costs = array([*args]).T
    is_efficient = ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = npall(npany(costs[:i] > c, axis=1)) and npall(
            npany(costs[i + 1 :] > c, axis=1)
        )

    return is_efficient

# Shell number calculation
# -------------------------
@make_symbolic
def stratum_min(*args, max_depth=10):
    r"""Compute Pareto stratum number

    Compute the Pareto stratum number for a given dataset.

    Args:
        xi (iterable OR gr.Intention()): Feature to minimize; use -X to maximize
        max_depth (int): Maximum depth for recursive computation; stratum numbers exceeding
            this value will not be computed and will be flagged as NaN.

    Returns:
        np.array of floats: Pareto stratum number

    References:
        del Rosario, Rupp, Kim, Antono, and Ling "Assessing the frontier: Active learning, model accuracy, and multi-objective candidate discovery and optimization" (2020) J. Chem. Phys.
    """
    # Check invariants
    lengths = map(len, args)
    if len(set(lengths)) > 1:
        raise ValueError("All arguments to stratum_min must be of equal length")

    # Set default as NaN
    costs = array([*args]).T
    n = costs.shape[0]
    stratum = ones(n)
    stratum[:] = NaN

    # Successive computation of stratum numbers
    active = ones(n, dtype=bool)
    idx_all = arange(n, dtype=int)

    i = 1
    while any(active) and (i <= max_depth):
        idx = idx_all[active]
        pareto = pareto_min(costs[idx].T)
        stratum[idx[pareto]] = i
        active[idx[pareto]] = False

        i += 1

    return stratum


# Factors
# -------------------------
@make_symbolic
def fct_reorder(f, x, fun=median):
    r"""Reorder a factor on another variable

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
    r"""Wrapper for pandas Series.fillna

    (See below for Pandas documentation)

    Examples:
        >>> import grama as gr
        >>> X = gr.Intention()
        >>> df = gr.df_make(x=[1, gr.NaN], y=[2, 3])
        >>> df_filled = (
        >>>     df
        >>>     >> gr.tf_mutate(x=gr.fillna(X.x, 0))
        >>> )
    """
    return Series.fillna(*args, **kwargs)


fillna.__doc__ = fillna.__doc__ + Series.fillna.__doc__
