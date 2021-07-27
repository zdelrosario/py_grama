__all__ = [
    "add_pipe",
    "copy_meta",
    "custom_formatwarning",
    "df_equal",
    "df_make",
    "param_dist",
    "pipe",
    "valid_dist",
    "tran_outer",
    "tf_outer",
]

import warnings
from functools import wraps
from inspect import signature
from numbers import Integral
from pandas import DataFrame, concat
from pandas.testing import assert_frame_equal
from scipy.stats import alpha, anglit, arcsine, argus, beta, betaprime, \
    bradford, burr, burr12, cauchy, chi, chi2, cosine, crystalball, dgamma, \
    dweibull, erlang, expon, exponnorm, exponweib, exponpow, f, fatiguelife, \
    fisk, foldcauchy, foldnorm, gaussian_kde, genlogistic, gennorm, genpareto, \
    genexpon, genextreme, gausshyper, gamma, gengamma, genhalflogistic, \
    gilbrat, gompertz, gumbel_r, gumbel_l, halfcauchy, halflogistic, \
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
    "gilbrat": gilbrat,
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
    "gilbrat": ["loc", "scale"],
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


## Safe length-checker
def safelen(x):
    try:
        return len(x)
    except TypeError:
        return 1


## DataFrame constructor utility
def df_make(**kwargs):
    r"""Construct a DataFrame

    Helper function to construct a DataFrame.

    Keyword Args:
        varname (iterable): Column for constructed dataframe; column
                            name inferred from variable name.
    Returns:
        DataFrame: Constructed DataFrame

    Preconditions:
        All provided iterables must have identical length or be of
        length one.

        All provided variable names (keyword arguments) must be distinct.

    Examples:
        A common use-case is to use df_make() to pass values to
        the df_det keyword argument succinctly;

        >>> import grama as gr
        >>> from models import make_test
        >>> md = make_test()
        >>> md >> \
        >>>     gr.ev_monte_carlo(
        >>>         n=1e3,
        >>>         df_det=gr.df_make(x2=[1, 2])
        >>>     )

    """
    ## Check lengths
    lengths = [safelen(v) for v in kwargs.values()]
    length_max = max(lengths)

    if not all([(l == length_max) | (l == 1) for l in lengths]):
        raise ValueError("Column lengths must be identical or one.")

    ## Construct dataframe
    df_res = DataFrame()
    for key in kwargs.keys():
        try:
            if len(kwargs[key]) > 1:
                df_res[key] = kwargs[key]
            else:
                df_res[key] = [kwargs[key][0]] * length_max
        except TypeError:
            df_res[key] = [kwargs[key]] * length_max

    return df_res


## DataFrame equality checker
def df_equal(df1, df2, close=False, precision=3):
    """Check DataFrame equality

    Check that two dataframes have the same columns and values. Allow column
    order to differ.

    Args:
        df1 (DataFrame): Comparison input 1
        df2 (DataFrame): Comparison input 2

    Returns:
        bool: Result of comparison

    """

    if not set(df1.columns) == set(df2.columns):
        return False

    if close:
        try:
            assert_frame_equal(
                df1[df2.columns],
                df2,
                check_dtype=False,
                check_exact=False,
                check_less_precise=precision,
            )
            return True
        except:
            return False
    else:
        return df1[df2.columns].equals(df2)


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

    Examples:
        >>> import grama as gr
        >>> import pandas as pd
        >>> df = pd.DataFrame(dict(x=[1,2]))
        >>> df_outer = pd.DataFrame(dict(y=[3,4]))
        >>> df_res = gr.tran_outer(df, df_outer)
        >>> df_res
        >>>    x  y
        >>> 0  1  3
        >>> 1  2  3
        >>> 2  1  4
        >>> 3  2  4

    """
    n_rows = df.shape[0]
    list_df = []

    for ind in range(df_outer.shape[0]):
        df_rep = concat([df_outer.iloc[[ind]]] * n_rows, ignore_index=True)
        list_df.append(concat((df, df_rep), axis=1))

    return concat(list_df, ignore_index=True)


tf_outer = add_pipe(tran_outer)
