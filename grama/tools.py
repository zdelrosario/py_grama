__all__ = [
    "continuous_fit",
    "copy_meta",
    "custom_formatwarning",
    "df_equal",
    "param_dist",
    "pipe",
    "valid_dist"
]

import pandas as pd
import warnings

from functools import wraps
from numbers import Integral

from scipy.stats import alpha, anglit, arcsine, argus, beta, betaprime
from scipy.stats import bradford, burr, burr12, cauchy, chi, chi2, cosine
from scipy.stats import crystalball, dgamma, dweibull, erlang, expon, exponnorm
from scipy.stats import exponweib, exponpow, f, fatiguelife, fisk, foldcauchy
from scipy.stats import foldnorm, frechet_r, frechet_l, genlogistic, gennorm
from scipy.stats import genpareto, genexpon, genextreme, gausshyper, gamma
from scipy.stats import gengamma, genhalflogistic, gilbrat, gompertz
from scipy.stats import gumbel_r, gumbel_l, halfcauchy, halflogistic, halfnorm
from scipy.stats import halfgennorm, hypsecant, invgamma, invgauss, invweibull
from scipy.stats import johnsonsb, johnsonsu, kappa4, kappa3, ksone, kstwobign
from scipy.stats import laplace, levy, levy_l, levy_stable, logistic, loggamma
from scipy.stats import loglaplace, lognorm, lomax, maxwell, mielke, moyal, nakagami
from scipy.stats import ncx2, ncf, nct, norm, norminvgauss, pareto, pearson3
from scipy.stats import powerlaw, powerlognorm, powernorm, rdist, rayleigh
from scipy.stats import rice, recipinvgauss, skewnorm, t, trapz, triang, truncexpon
from scipy.stats import truncnorm, tukeylambda, uniform, vonmises, vonmises_line
from scipy.stats import wald, weibull_min, weibull_max, wrapcauchy

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
    "frechet_r": frechet_r,
    "frechet_l": frechet_l,
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
    "wrapcauchy": wrapcauchy
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
        df_source (DataFrame):
        df_target (DataFrame):

    Returns:
        DataFrame: df_target with copied metadata
    """
    df_target._grouped_by = getattr(df_source, '_grouped_by', None)
    df_target._plot_info  = getattr(df_source, '_plot_info', None)
    df_target._meta       = getattr(df_source, '_meta', None)

    return df_target

## Pipe decorator
class pipe(object):
    __name__ = "pipe"

    def __init__(self, function):
        # @wraps(function) # Preserve documentation?

        self.function = function
        self.__doc__ = function.__doc__

        self.chained_pipes = []

    def __rshift__(self, other):
        assert isinstance(other, pipe)
        self.chained_pipes.append(other)
        return self

    def __rrshift__(self, other):
        other_copy = other.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(other, pd.DataFrame):
                # other_copy._grouped_by = getattr(other, '_grouped_by', None)
                # other_copy._plot_info = getattr(other, '_plot_info', None)
                # other_copy._meta = getattr(other, '_meta', None)
                other_copy = copy_meta(other, other_copy)

        result = self.function(other_copy)

        for p in self.chained_pipes:
            result = p.__rrshift__(result)
        return result

    def __repr__(self):
        return self.__name__ + ": " + self.function.__name__ + \
            "\n Did you mean to use a full-prefix verb?"

    def __call__(self, *args, **kwargs):
        return pipe(lambda x: self.function(x, *args, **kwargs))

## DataFrame equality checker
def df_equal(df1, df2, close=False):
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
            pd.testing.assert_frame_equal(
                df1[df2.columns],
                df2,
                check_dtype=False,
                check_exact=False
            )
            return True
        except:
            return False
    else:
        return df1[df2.columns].equals(df2)

## Fit a named scipy.stats distribution
def continuous_fit(data, dist, name=True, sign=None):
    """Fit scipy.stats continuous distirbution

    Fits a named scipy.stats continuous distribution. Intended to be used to
    define a marginal distribution from data.

    Args:
        data (iterable): Data for fit
        dist (str): Distribution to fit
        name (bool): Include distribution name?
        sign (bool): Include sign? (Optional)

    Returns:
        dict: Distribution parameters organized by keyword

    Examples:

        >>> import grama as gr
        >>> from grama.data import df_stang
        >>> param_E  = gr.continuous_fit(df_stang.E, "norm")
        >>> param_mu = gr.continuous_fit(df_stang.mu, "beta")
        >>> md = gr.Model("Marginal Example") >> \
        >>>     gr.cp_marginals(E=param_E, mu=param_mu)
        >>> md.printpretty()

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = valid_dist[dist].fit(data)

    res = dict(zip(param_dist[dist], res))

    if name:
        res["dist"] = dist

    if sign is not None:
        if not (sign in [-1, 0, +1]):
            raise ValueError("Invalid `sign`")
        res["sign"] = sign

    return res

## Monkey-patched warning fcn
def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return "Warning: " + str(msg) + '\n'
