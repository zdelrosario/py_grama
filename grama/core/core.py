## grama core functions
# Zachary del Rosario, March 2019

__all__ = [
    "pipe",
    "domain_",
    "density_",
    "model_",
    "model_vectorized_",

    "eval_df",
    "ev_df"
]

import numpy as np
import pandas as pd
import warnings

from toolz import curry
from numpy.linalg import cholesky

from scipy.stats import alpha, beta, chi, chi2, expon, gamma, laplace
from scipy.stats import ncf, nct, pareto, powerlaw, rayleigh
from scipy.stats import t, truncexpon, truncnorm, uniform, weibull_min, weibull_max
from scipy.stats import norm, lognorm

valid_dist = {
    "alpha"       : alpha,
    "beta"        : beta,
    "chi"         : chi,
    "chi2"        : chi2,
    "expon"       : expon,
    "gamma"       : gamma,
    "laplace"     : laplace,
    "ncf"         : ncf,
    "nct"         : nct,
    "pareto"      : pareto,
    "powerlaw"    : powerlaw,
    "rayleigh"    : rayleigh,
    "t"           : t,
    "truncexpon"  : truncexpon,
    "truncnorm"   : truncnorm,
    "uniform"     : uniform,
    "weibull_min" : weibull_min,
    "weibull_max" : weibull_max,
    "norm"        : norm,
    "lognorm"     : lognorm
}

## Pipe decorator
class pipe(object):
    __name__ = "pipe"

    def __init__(self, function):
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
                other_copy._grouped_by = getattr(other, '_grouped_by', None)

        result = self.function(other_copy)

        for p in self.chained_pipes:
            result = p.__rrshift__(result)
        return result

    def __call__(self, *args, **kwargs):
        return pipe(lambda x: self.function(x, *args, **kwargs))

## Core functions
##################################################
# Domain parent class
class domain_:
    """Parent class for input domains
    """
    def __init__(
            self,
            hypercube = True,
            inputs    = ["x"],
            bounds    = {"x": [-1., +1.]},
            feasible  = lambda x: (-1 <= x) * (x <= +1)
    ):
        """Initialize

        @param hypercube bool flag
        @param inputs list of input names
        """
        self.hypercube = hypercube if (hypercube is not None) else True
        self.inputs    = inputs if (inputs is not None) else ["x"]
        self.bounds    = bounds if (bounds is not None) else {"x": [-1., +1.]}
        self.feasible  = feasible if (feasible is not None) else lambda x: (-1<=x) * (x<=+1)

# Density parent class
class density_:
    """Parent class for joint densities
    """
    def __init__(
            self,
            pdf         = None,
            pdf_factors = None,
            pdf_param   = None,
            pdf_corr    = None,
            pdf_qt_sign = None
    ):
        """Initialize

        @param pdf density function \rho(x) : R^n_in -> R
        @param pdf_factors if joint density can be factored, list of names
               of marginal distributions
        @param pdf_param if joint density can be factored, list of dict
               of marginal density parameters
        @param pdf_corr correlation matrix for copula representation,
               either None (for independent densities) or a list of
               correlation entries ordered as np.triu_indices(n_in, 1)
        @param pdf_qt_sign array of integers in [-1, 0, +1] used to indicate
               the "conservative" direction for each input (if any).
                 -1: Small values are conservative
                 +1: Large values are conservative
                  0: Use the median
               Useful for "conservative" quantile evaluation approaches.

        @pre (len(pdf_factors) == n_in) || (pdf_factors is None)
        @pre (len(pdf_param) == n_in) || (pdf_param is None)
        @pre (len(pdf_corr == len(np.triu_indices(n_in, 1)[0]))) || (pdf_param is None)
        @pre (len(pdf_qt_sign) == n_in) || (pdf_qt_sign is None)
        """
        self.pdf         = pdf if (pdf is not None) else lambda x: 0.5
        self.pdf_factors = pdf_factors if (pdf_factors is not None) else ["uniform"]
        self.pdf_param   = pdf_param if (pdf_param is not None) else [
            {"loc": -1., "scale": +2.}
        ]
        self.pdf_corr    = pdf_corr if (pdf_corr is not None) else None
        self.pdf_qt_sign = pdf_qt_sign if (pdf_qt_sign is not None) else [0] * len(self.pdf_factors)

# Model parent class
class model_:
    """Parent class for grama models.
    """

    def __init__(
            self,
            name     = None,
            function = None,
            outputs  = None,
            domain   = None,
            density  = None,
    ):
        """Constructor

        @param function defining the model mapping f(x) : R^n_in -> R^n_out
        @param inputs to function; ordering of abstract inputs x given by inputs
        @param ouputs of function outputs or None
        @param domain object of class domain_ or None
        @param density object of class density_ or None

        @pre len(domain.inputs) == n_in
        @pre len(outputs) == n_out
        @pre isinstance(domain, domain_)
        @pre isinstance(density, density_) || (density is None)

        Default model is 1D identity over the interval [-1, +1] with a uniform density.
        """
        self.name     = name if (name is not None) else "Default"
        self.function = function if (function is not None) else lambda x: x
        self.outputs  = outputs if (outputs is not None) else ["f"]
        self.domain   = domain if (domain is not None) else domain_()
        self.density  = density if (density is not None) else density_()

        ## Convenience constants
        self.n_in  = len(self.domain.inputs)
        self.n_out = len(self.outputs)

    def evaluate(self, df):
        """Evaluate function using an input dataframe

        Does not assume a vectorized function.
        """
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.domain.inputs).issubset(set(df.columns)):
            raise ValueError("Model inputs not a subset of given columns")

        ## Set up output
        n_rows  = df.shape[0]
        results = np.zeros((n_rows, self.n_out))
        for ind in range(n_rows):
            results[ind] = self.function(df.loc[ind, self.domain.inputs])

        ## Package output as DataFrame
        return pd.DataFrame(data=results, columns=self.outputs)

    def sample_quantile(self, quantiles):
        """Convert quantiles to input samples

        @pre quantiles.shape[1] == n_in
        """
        samples = np.zeros(quantiles.shape)

        ## Perform copula conversion, if necessary
        if self.density.pdf_corr is not None:
            ## Build correlation structure
            Sigma                                = np.eye(self.n_in)
            Sigma[np.triu_indices(self.n_in, 1)] = self.density.pdf_corr
            Sigma                                = Sigma + (Sigma - np.eye(self.n_in)).T
            Sigma_h                              = cholesky(Sigma)
            ## Convert samples
            gaussian_samples = np.dot(norm.ppf(quantiles), Sigma_h.T)

            ## Convert to uniform marginals
            quantiles = norm.cdf(gaussian_samples)
        ## Skip if no dependence structure

        ## Apply appropriate marginal
        for ind in range(self.n_in):
            ## Map with inverse density
            samples[:, ind] = valid_dist[self.density.pdf_factors[ind]].ppf(
                quantiles[:, ind],
                **self.density.pdf_param[ind]
            )
        return samples

    def name_corr(self):
        """Name the correlation elements
        """
        ## Build matrix of names
        corr_mat = []
        for ind in range(self.n_in):
            corr_mat.append(
                list(map(
                    lambda s: s + "," + self.domain.inputs[ind],
                    self.domain.inputs
                ))
            )

        ## Access matrix of names
        corr_names = dict()
        corr_ind = np.triu_indices(self.n_in, 1)
        for knd in range(len(corr_ind[0])):
            ind = corr_ind[0][knd]
            jnd = corr_ind[1][knd]
            corr_names["corr_" + str(knd)] = corr_mat[ind][jnd]

        return corr_names

    def copy(self):
        """Make a copy of this model
        """
        model = model_(
            name     = self.name,
            function = self.function,
            outputs  = self.outputs,
            domain   = self.domain,
            density  = self.density
        )
        return model

    def printpretty(self):
        """Formatted print of model attributes
        """
        print(self.name)
        print("  inputs  = {}".format(self.domain.inputs))
        print("  outputs = {}".format(self.outputs))

# Derived dataframe-vectorized model
class model_vectorized_(model_):
    """Derived class for grama models.

    Given function must be vectorized over dataframes
    """

    def evaluate(self, df):
        """Evaluate function using an input dataframe

        Assumes function is vectorized over dataframes.
        """
        return self.function(df)

    def copy(self):
        """Make a copy of this model
        """
        model = model_vectorized_(
            name     = self.name,
            function = self.function,
            outputs  = self.outputs,
            domain   = self.domain,
            density  = self.density
        )
        return model

## Default evaluation function
# --------------------------------------------------
@curry
def eval_df(model, df=None, append=True):
    """Evaluates a given model at a given dataframe

    @param df input dataframe to evaluate (Pandas.DataFrame)
    @param append bool flag; append results to original dataframe?
    """

    if df is None:
        raise ValueError("No input df given!")

    df_res = model.evaluate(df)

    if append:
        df_res = pd.concat([df.reset_index(drop=True), df_res], axis=1)

    return df_res

@pipe
def ev_df(*args, **kwargs):
    return eval_df(*args, **kwargs)
