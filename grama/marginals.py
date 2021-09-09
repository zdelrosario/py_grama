__all__ = [
    "marg_gkde",
    "marg_named",
    "Marginal",
    "MarginalNamed",
    "MarginalGKDE",
    "param_dist",
    "valid_dist",
]

import copy
import warnings
from abc import ABC, abstractmethod
from numpy import zeros, array, Inf
from numpy import min as npmin
from numpy import max as npmax
from pandas import DataFrame
from scipy.optimize import root_scalar
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

## Marginal classes
##################################################
# Marginal parent class
class Marginal(ABC):
    """Parent class for marginal distributions
    """

    def __init__(self, sign=0):
        self.sign = sign

    @abstractmethod
    def copy(self):
        pass

    ## Fitting function
    @abstractmethod
    def fit(self, data):
        pass

    ## Likelihood function
    @abstractmethod
    def l(self, x):
        pass

    ## Cumulative density function
    @abstractmethod
    def p(self, x):
        pass

    ## Quantile function
    @abstractmethod
    def q(self, p):
        pass

    ## Summary
    @abstractmethod
    def summary(self):
        pass


## Named marginal class
class MarginalNamed(Marginal):
    """Marginal using a named distribution from gr.valid_dist"""

    def __init__(self, d_name=None, d_param=None, **kw):
        super().__init__(**kw)

        self.d_name = d_name
        self.d_param = d_param

    def copy(self):
        new_marginal = MarginalNamed(
            sign=self.sign, d_name=self.d_name, d_param=copy.deepcopy(self.d_param)
        )

        return new_marginal

    ## Fitting function
    def fit(self, data):
        param = valid_dist[self.d_name].fit(data)
        self.d_param = dict(zip(param_dist[dist], param))

    ## Likelihood function
    def l(self, x):
        return valid_dist[self.d_name].pdf(x, **self.d_param)

    ## Cumulative density function
    def p(self, x):
        return valid_dist[self.d_name].cdf(x, **self.d_param)

    ## Quantile function
    def q(self, p):
        return valid_dist[self.d_name].ppf(p, **self.d_param)

    ## Summary
    def summary(self):
        return "({0:+}) {1:}, {2:}".format(self.sign, self.d_name, self.d_param)


## Gaussian KDE marginal class
class MarginalGKDE(Marginal):
    """Marginal using scipy.stats.gaussian_kde"""

    def __init__(self, kde, atol=1e-6, **kw):
        super().__init__(**kw)

        self.kde = kde
        self.atol = atol
        self._set_bracket()

    def copy(self):
        new_marginal = MarginalGKDE(kde=copy.deepcopy(self.kde), atol=self.atol,)

        return new_marginal

    def _set_bracket(self):
        ## Calibrate the quantile brackets based on desired accuracy
        bracket = [npmin(self.kde.dataset), npmax(self.kde.dataset)]

        sol_lo = root_scalar(
            lambda x: self.atol - self.p(x),
            x0=bracket[0],
            x1=bracket[1],
            method="secant",
        )
        sol_hi = root_scalar(
            lambda x: 1 - self.atol - self.p(x),
            x0=bracket[0],
            x1=bracket[1],
            method="secant",
        )

        self.bracket = [sol_lo.root, sol_hi.root]

    ## Fitting function
    def fit(self, data):
        self.kde = gaussian_kde(data)
        self._set_bracket()

    ## Likelihood function
    def l(self, x):
        return self.kde.pdf(x)

    ## Cumulative density function
    def p(self, x):
        try:
            return array([self.kde.integrate_box_1d(-Inf, v) for v in x])
        except TypeError:
            return self.kde.integrate_box_1d(-Inf, x)

    ## Quantile function
    def q(self, p):
        p_bnd = self.p(self.bracket)

        ## Create scalar solver
        def qscalar(val):
            if val <= p_bnd[0]:
                return self.bracket[0]
            if val >= p_bnd[1]:
                return self.bracket[1]
            sol = root_scalar(
                lambda x: val - self.p(x),
                bracket=self.bracket,
                method="bisect",
                xtol=self.atol,
            )
            return sol.root

        ## Iterate over all given values
        try:
            res = zeros(len(p))
            for i in range(len(p)):
                res[i] = qscalar(p[i])
        except TypeError:
            res = qscalar(p)

        return res

    ## Summary
    def summary(self):
        p_bnd = self.p(self.bracket)

        return "({0:+}) gaussian KDE, n={1:2.1f}/{2:}, f={3:2.1e}, ".format(
            self.sign, self.kde.neff, self.kde.dataset.shape[1], self.kde.factor
        ) + "b=[{0:2.1e}, {1:2.1e}], a={2:1.0e}".format(
            self.bracket[0], self.bracket[1], self.atol
        )

## Marginal functions
##################################################
## Fit a named scipy.stats distribution
def marg_named(data, dist, name=True, sign=None):
    r"""Fit scipy.stats continuous distirbution

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
        >>>     gr.cp_marginals(
        >>>         E=gr.marg_named(df_stang.E, "norm"),
        >>>         mu=gr.marg_named(df_stang.mu, "beta")
        >>>     )
        >>> md.printpretty()

    """
    ## Catch case where user provides entire DataFrame
    if isinstance(data, DataFrame):
        raise ValueError("`data` argument must be a single column; try data.var")

    ## Fit the distribution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        param = valid_dist[dist].fit(data)

    param = dict(zip(param_dist[dist], param))

    if sign is not None:
        if not (sign in [-1, 0, +1]):
            raise ValueError("Invalid `sign`")
    else:
        sign = 0

    return MarginalNamed(sign=sign, d_name=dist, d_param=param)


## Fit a gaussian kernel density estimate (KDE) to data
def marg_gkde(data, sign=None):
    r"""Fit a gaussian KDE to data

    Fits a gaussian kernel density estimate (KDE) to data.

    Args:
        data (iterable): Data for fit
        sign (bool): Include sign? (Optional)

    Returns:
        gr.MarginalGKDE: Marginal distribution

    Examples:

        >>> import grama as gr
        >>> from grama.data import df_stang
        >>> md = gr.Model("Marginal Example") >> \
        >>>     gr.cp_marginals(
        >>>         E=gr.marg_gkde(df_stang.E),
        >>>         mu=gr.marg_gkde(df_stang.mu)
        >>>     )
        >>> md.printpretty()

    """
    ## Catch case where user provides entire DataFrame
    if isinstance(data, DataFrame):
        raise ValueError("`data` argument must be a single column; try data.var")

    kde = gaussian_kde(data)
    if sign is not None:
        if not (sign in [-1, 0, +1]):
            raise ValueError("Invalid `sign`")
    else:
        sign = 0

    return MarginalGKDE(kde, sign=sign)
