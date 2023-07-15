__all__ = [
    "add_pipe",
    "copy_meta",
    "custom_formatwarning",
    "lookup",
    "hide_traceback",
    "param_dist",
    "pipe",
    "valid_dist",
    "tran_outer",
    "tf_outer",
]

import warnings
import sys
from functools import wraps
from inspect import signature
from numbers import Integral
from numpy import empty
from pandas import DataFrame, concat
from pandas.core.dtypes.common import is_object_dtype
from pandas._libs import (
    algos as libalgos,
    lib,
    properties,
)
from pandas.testing import assert_frame_equal
from scipy.stats import alpha, anglit, arcsine, argus, beta, betaprime, \
    bradford, burr, burr12, cauchy, chi, chi2, cosine, crystalball, dgamma, \
    dweibull, erlang, expon, exponnorm, exponweib, exponpow, f, fatiguelife, \
    fisk, foldcauchy, foldnorm, gaussian_kde, genlogistic, gennorm, genpareto, \
    genexpon, genextreme, gausshyper, gamma, gengamma, genhalflogistic, \
    gibrat, gompertz, gumbel_r, gumbel_l, halfcauchy, halflogistic, \
    halfnorm, halfgennorm, hypsecant, invgamma, invgauss, invweibull, \
    johnsonsb, johnsonsu, kappa4, kappa3, ksone, kstwobign, laplace, levy, \
    levy_l, levy_stable, logistic, loggamma, loglaplace, lognorm, lomax, \
    maxwell, mielke, moyal, nakagami, ncx2, ncf, nct, norm, norminvgauss, \
    pareto, pearson3, powerlaw, powerlognorm, powernorm, rdist, rayleigh, \
    rice, recipinvgauss, skewnorm, t, trapz, triang, truncexpon, truncnorm, \
    tukeylambda, uniform, vonmises, vonmises_line, wald, weibull_min, \
    weibull_max, wrapcauchy
from toolz import curry


## Scipy metadata
valid_dist = {
    "alpha": alpha,
    "anglit": anglit,
    "arcsine": arcsine,
    "argus": argus,
    "beta": beta,
    "betaprime": betaprime,
    "bradford": bradford,
    "burr": burr,
    "burr12": burr12,
    "cauchy": cauchy,
    "chi": chi,
    "chi2": chi2,
    "cosine": cosine,
    "crystalball": crystalball,
    "dgamma": dgamma,
    "dweibull": dweibull,
    "erlang": erlang,
    "expon": expon,
    "exponnorm": exponnorm,
    "exponweib": exponweib,
    "exponpow": exponpow,
    "f": f,
    "fatiguelife": fatiguelife,
    "fisk": fisk,
    "foldcauchy": foldcauchy,
    "foldnorm": foldnorm,
    # "frechet_r": frechet_r,
    # "frechet_l": frechet_l,
    "genlogistic": genlogistic,
    "gennorm": gennorm,
    "genpareto": genpareto,
    "genexpon": genexpon,
    "genextreme": genextreme,
    "gausshyper": gausshyper,
    "gamma": gamma,
    "gengamma": gengamma,
    "genhalflogistic": genhalflogistic,
    # "geninvgauss": geninvgauss,
    "gibrat": gibrat,
    "gompertz": gompertz,
    "gumbel_r": gumbel_r,
    "gumbel_l": gumbel_l,
    "halfcauchy": halfcauchy,
    "halflogistic": halflogistic,
    "halfnorm": halfnorm,
    "halfgennorm": halfgennorm,
    "hypsecant": hypsecant,
    "invgamma": invgamma,
    "invgauss": invgauss,
    "invweibull": invweibull,
    "johnsonsb": johnsonsb,
    "johnsonsu": johnsonsu,
    "kappa4": kappa4,
    "kappa3": kappa3,
    "ksone": ksone,
    "kstwobign": kstwobign,
    "laplace": laplace,
    "levy": levy,
    "levy_l": levy_l,
    "levy_stable": levy_stable,
    "logistic": logistic,
    "loggamma": loggamma,
    "loglaplace": loglaplace,
    "lognorm": lognorm,
    # "loguniform": loguniform,
    "lomax": lomax,
    "maxwell": maxwell,
    "mielke": mielke,
    "moyal": moyal,
    "nakagami": nakagami,
    "ncx2": ncx2,
    "ncf": ncf,
    "nct": nct,
    "norm": norm,
    "norminvgauss": norminvgauss,
    "pareto": pareto,
    "pearson3": pearson3,
    "powerlaw": powerlaw,
    "powerlognorm": powerlognorm,
    "powernorm": powernorm,
    "rdist": rdist,
    "rayleigh": rayleigh,
    "rice": rice,
    "recipinvgauss": recipinvgauss,
    "skewnorm": skewnorm,
    "t": t,
    "trapz": trapz,
    "triang": triang,
    "truncexpon": truncexpon,
    "truncnorm": truncnorm,
    "tukeylambda": tukeylambda,
    "uniform": uniform,
    "vonmises": vonmises,
    "vonmises_line": vonmises_line,
    "wald": wald,
    "weibull_min": weibull_min,
    "weibull_max": weibull_max,
    "wrapcauchy": wrapcauchy,
}

param_dist = {
    "alpha": ["a", "loc", "scale"],
    "anglit": ["loc", "scale"],
    "arcsine": ["loc", "scale"],
    "argus": ["chi", "loc", "scale"],
    "beta": ["a", "b", "loc", "scale"],
    "betaprime": ["a", "b", "loc", "scale"],
    "bradford": ["c", "loc", "scale"],
    "burr": ["c", "d", "loc", "scale"],
    "burr12": ["c", "d", "loc", "scale"],
    "cauchy": ["loc", "scale"],
    "chi": ["df", "loc", "scale"],
    "chi2": ["df", "loc", "scale"],
    "cosine": ["loc", "scale"],
    "crystalball": ["beta", "m", "loc", "scale"],
    "dgamma": ["a", "loc", "scale"],
    "dweibull": ["c", "loc", "scale"],
    "erlang": ["loc", "scale"],
    "expon": ["loc", "scale"],
    "exponnorm": ["K", "loc", "scale"],
    "exponweib": ["a", "c", "loc", "scale"],
    "exponpow": ["b", "loc", "scale"],
    "f": ["dfn", "dfd", "loc", "scale"],
    "fatiguelife": ["c", "loc", "scale"],
    "fisk": ["c", "loc", "scale"],
    "foldcauchy": ["c", "loc", "scale"],
    "foldnorm": ["c", "loc", "scale"],
    "frechet_r": ["c", "loc", "scale"],
    "frechet_l": ["c", "loc", "scale"],
    "genlogistic": ["c", "loc", "scale"],
    "gennorm": ["beta", "loc", "scale"],
    "genpareto": ["c", "loc", "scale"],
    "genexpon": ["a", "b", "c", "loc", "scale"],
    "genextreme": ["c", "loc", "scale"],
    "gausshyper": ["a", "b", "c", "z", "loc", "scale"],
    "gamma": ["a", "loc", "scale"],
    "gengamma": ["a", "c", "loc", "scale"],
    "genhalflogistic": ["c", "loc", "scale"],
    "geninvgauss": ["p", "b", "loc", "scale"],
    "gibrat": ["loc", "scale"],
    "gompertz": ["c", "loc", "scale"],
    "gumbel_r": ["loc", "scale"],
    "gumbel_l": ["loc", "scale"],
    "halfcauchy": ["loc", "scale"],
    "halflogistic": ["loc", "scale"],
    "halfnorm": ["loc", "scale"],
    "halfgennorm": ["beta", "loc", "scale"],
    "hypsecant": ["loc", "scale"],
    "invgamma": ["a", "loc", "scale"],
    "invgauss": ["mu", "loc", "scale"],
    "invweibull": ["c", "loc", "scale"],
    "johnsonsb": ["a", "b", "loc", "scale"],
    "johnsonsu": ["a", "b", "loc", "scale"],
    "kappa4": ["h", "k", "loc", "scale"],
    "kappa3": ["a", "loc", "scale"],
    "ksone": ["n", "loc", "scale"],
    "kstwobign": ["loc", "scale"],
    "laplace": ["loc", "scale"],
    "levy": ["loc", "scale"],
    "levy_l": ["loc", "scale"],
    "levy_stable": ["alpha", "beta", "loc", "scale"],
    "logistic": ["loc", "scale"],
    "loggamma": ["c", "loc", "scale"],
    "loglaplace": ["c", "loc", "scale"],
    "lognorm": ["s", "loc", "scale"],
    "loguniform": ["a", "b", "loc", "scale"],
    "lomax": ["c", "loc", "scale"],
    "maxwell": ["loc", "scale"],
    "mielke": ["k", "s", "loc", "scale"],
    "moyal": ["loc", "scale"],
    "nakagami": ["nu", "loc", "scale"],
    "ncx2": ["df", "nc", "loc", "scale"],
    "ncf": ["dfn", "dfd", "nc", "loc", "scale"],
    "nct": ["df", "nc", "loc", "scale"],
    "norm": ["loc", "scale"],
    "norminvgauss": ["a", "b", "loc", "scale"],
    "pareto": ["b", "loc", "scale"],
    "pearson3": ["skew", "loc", "scale"],
    "powerlaw": ["a", "loc", "scale"],
    "powerlognorm": ["c", "s", "loc", "scale"],
    "powernorm": ["c", "loc", "scale"],
    "rdist": ["c", "loc", "scale"],
    "rayleigh": ["loc", "scale"],
    "rice": ["b", "loc", "scale"],
    "recipinvgauss": ["mu", "loc", "scale"],
    "semicircular": ["loc", "scale"],
    "skewnorm": ["a", "loc", "scale"],
    "t": ["df", "loc", "scale"],
    "trapz": ["c", "d", "loc", "scale"],
    "triang": ["c", "loc", "scale"],
    "truncexpon": ["b", "loc", "scale"],
    "truncnorm": ["a", "b", "loc", "scale"],
    "tukeylambda": ["lam", "loc", "scale"],
    "uniform": ["loc", "scale"],
    "vonmises": ["kappa", "loc", "scale"],
    "vonmises_line": ["kappa", "loc", "scale"],
    "wald": ["loc", "scale"],
    "weibull_min": ["c", "loc", "scale"],
    "weibull_max": ["c", "loc", "scale"],
    "wrapcauchy": ["c", "loc", "scale"],
}


## Core helper functions
##################################################
## Metadata copy
def copy_meta(df_source, df_target):
    """Internal metadata copy tool

    Args:
        df_source (DataFrame): Original dataframe
        df_target (DataFrame): Target dataframe; receives metadata

    Returns:
        DataFrame: df_target with copied metadata
    """
    df_target._grouped_by = getattr(df_source, "_grouped_by", None)
    df_target._plot_info = getattr(df_source, "_plot_info", None)
    df_target._meta = getattr(df_source, "_meta", None)

    return df_target


## Pipe decorator
class pipe(object):
    __name__ = "pipe"

    def __init__(self, function):
        self.function = function
        # self.__doc__ = function.__doc__
        self.chained_pipes = []

    def __rshift__(self, other):
        assert isinstance(other, pipe)
        self.chained_pipes.append(other)
        return self

    def __rrshift__(self, other):
        other_copy = other.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(other, DataFrame):
                other_copy = copy_meta(other, other_copy)

        result = self.function(other_copy)

        for p in self.chained_pipes:
            result = p.__rrshift__(result)
        return result

    def __repr__(self):
        return (
            self.__name__
            + ": "
            + self.function.__name__
            + "\n Did you mean to use a full-prefix verb?"
        )

    def __call__(self, *args, **kwargs):
        return pipe(lambda x: self.function(x, *args, **kwargs))


## Pipe applicator
def add_pipe(fun):
    class NewPipe(pipe):
        __name__ = fun.__name__
        __doc__ = (
            fun.__doc__
            #"Pipe-enabled version of {}\n".format(fun)
            #+ "Inherited Signature: {}\n".format(signature(fun))
            #+ fun.__doc__
        )

    return NewPipe(fun)


## Monkey-patched warning fcn
def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return "Warning: " + str(msg) + "\n"


## DataFrame outer product
# --------------------------------------------------
@curry
def tran_outer(df, df_outer):
    r"""Outer merge

    Perform an outer-merge on two dataframes.

    Args:
        df (DataFrame): Data to merge
        df_outer (DataFrame): Data to merge; outer

    Returns:
        DataFrame: Merged data

    Examples::

        import grama as gr
        import pandas as pd
        df = pd.DataFrame(dict(x=[1,2]))
        df_outer = pd.DataFrame(dict(y=[3,4]))
        df_res = gr.tran_outer(df, df_outer)
        df_res
             x  y
          0  1  3
          1  2  3
          2  1  4
          3  2  4

    """
    # Check invariants
    if (df.shape[0] == 0) and (df_outer.shape[0] == 0):
        raise ValueError("At least one of df and df_outer must be non-empty")
    # Handle single-empty cases
    if (df.shape[0] == 0):
        return df_outer
    if (df_outer.shape[0] == 0):
        return df

    n_rows = df.shape[0]
    list_df = []

    for ind in range(df_outer.shape[0]):
        df_rep = concat([df_outer.iloc[[ind]]] * n_rows, ignore_index=True)
        list_df.append(concat((df, df_rep), axis=1))

    return concat(list_df, ignore_index=True)


tf_outer = add_pipe(tran_outer)


def lookup(df, row_labels, col_labels):
    r"""2D lookup function for a dataframe
    (Old Pandas Lookup Method)

    Args:
        df (DataFrame): DataFrame for lookup
        row_labels (List): Row labels to use for lookup.
        col_labels (List): Column labels to use for lookup.

    Returns:
        numpy.ndarray: Found values
    """
    n = len(row_labels)
    if n != len(col_labels):
        raise ValueError("Row labels must have same size as column labels")
    if not (df.index.is_unique and df.columns.is_unique):
        raise ValueError("DataFrame.lookup requires unique index and columns")

    thresh = 1000
    if not df._is_mixed_type or n > thresh:
        values = df.values
        ridx = df.index.get_indexer(row_labels)
        cidx = df.columns.get_indexer(col_labels)
        if (ridx == -1).any():
            raise KeyError("One or more row labels was not found")
        if (cidx == -1).any():
            raise KeyError("One or more column labels was not found")
        flat_index = ridx * len(df.columns) + cidx
        result = values.flat[flat_index]
    else:
        result = empty(n, dtype="O")
        for i, (r, c) in enumerate(zip(row_labels, col_labels)):
            print(r,c)
            result[i] = df._get_value(r, c)

    if is_object_dtype(result):
        result = lib.maybe_convert_objects(result)

    return result


# Suppress traceback
def hide_traceback():
    r"""Configure Jupyter to hide error traceback
    """
    ipython = get_ipython()

    def _hide_traceback(exc_tuple=None, filename=None, tb_offset=None,
                        exception_only=False, running_compiled_code=False):
        etype, value, tb = sys.exc_info()
        value.__cause__ = None  # suppress chained exceptions
        return ipython._showtraceback(etype, value, ipython.InteractiveTB.get_exception_only(etype, value))

    ipython.showtraceback = _hide_traceback
