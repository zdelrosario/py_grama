## grama core functions
# Zachary del Rosario, March 2019

__all__ = [
    "domain",
    "density",
    "model",
    "model_vectorized",
    "marginal_named",
    "valid_dist"
]

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..transforms import tran_outer
from ..tools import pipe

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

## Core functions
##################################################
# Domain parent class
class domain:
    """Parent class for input domains

    The domain is defined for all the variables; therefore it
    is the domain of the model's function.
    """
    def __init__(self, bounds=None, feasible=None):
        """Initialize

        @param bounds OrderedDict of variable bounds
        @param feasible vectorized function mapping variables to bool

        @pre isinstance(bounds, collection.OrderedDict)
        @pre (feasible is function) | (feasible is None)
        @pre len(bounds) == n where model.function:R^n -> R^m
        """
        self._bounds    = bounds
        self._feasible  = feasible
        self._variables = list(self._bounds.keys())

# Marginal parent class
class marginal_(ABC):
    """Parent class for marginal distributions
    """
    def __init__(self, var, sign=0):
        self._var = var
        self._sign = sign

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
class marginal_named(marginal_):
    """Marginal using a named distribution from core.valid_dist"""

    def __init__(self, var, d_name=None, d_param=None, **kw):
        super().__init__(var, **kw)

        self._d_name = d_name
        self._d_param = d_param

    ## Likelihood function
    def l(self, x):
        return valid_dist[self._d_name].pdf(x, **self._d_param)

    ## Cumulative density function
    def p(self, x):
        return valid_dist[self._d_name].cdf(x, **self._d_param)

    ## Quantile function
    def q(self, p):
        return valid_dist[self._d_name].ppf(p, **self._d_param)

    ## Summary
    def summary(self):
        return "{0:} ({1:}): {2:}, {3:}".format(
            self._var,
            self._sign,
            self._d_name,
            self._d_param
        )

# Density parent class
class density:
    """Parent class for joint densities

    The density is defined for all the random variables; therefore
    it explicitly defines the list of random variables, and together
    with the domain defines the deterministic variables via
    domain._variables - density._variables
    """
    def __init__(self, marginals=None, copula=None):
        """Initialize

        @param marginals list of marginal_ defining random variables
        @param copula TODO
        """
        self._marginals = marginals
        self._copula = copula
        self._var_rand = list(map(lambda m: m._var, self._marginals))

# Model parent class
class model:
    """Parent class for grama models.
    """

    def __init__(
            self,
            name=None,
            function=None,
            outputs=None,
            domain=None,
            density=None,
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
        """
        self.name     = name
        self.function = function
        self.outputs  = outputs
        self.domain   = domain
        self.density  = density

        self.update()

    def update(self):
        """Update model public attributes based on domain and
        density representation
        """
        ## Maintain list of variables and parameters
        self.var      = self.domain._variables
        self.var_rand = self.density._var_rand
        self.var_det  = list(set(self.var).difference(self.var_rand))

        ## TODO parameters

        ## Convenience constants
        self.n_var      = len(self.var)
        self.n_var_rand = len(self.var_rand)
        self.n_var_det  = len(self.var_det)
        self.n_out      = len(self.outputs)

    def det_nom(self):
        """Return nominal conditions for deterministic variables
        """
        df_nom = pd.DataFrame(
            data={
                var: [
                    0.5 * (
                        self.domain._bounds[var][0] + \
                        self.domain._bounds[var][1]
                    )
                ] for var in self.var_det
            }
        )
        return df_nom

    def evaluate_df(self, df):
        """Evaluate function using an input dataframe

        Does not assume a vectorized function.
        """
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.domain._variables).issubset(set(df.columns)):
            raise ValueError("Model inputs not a subset of given columns")

        ## Set up output
        n_rows  = df.shape[0]
        results = np.zeros((n_rows, self.n_out))
        for ind in range(n_rows):
            results[ind] = self.function(df.loc[ind, self.domain._variables])

        ## Package output as DataFrame
        return pd.DataFrame(data=results, columns=self.outputs)

    def var_outer(self, df_rand, df_det=None):
        """Constuct outer product of random and deterministic samples.

        @param df_rand DataFrame random variable samples
        @param df_det DataFrame deterministic variable samples
                      set to "nom" for nominal evaluation
        """
        ## Pass-through if no var_det
        if self.n_var_det == 0:
            return df_rand

        ## Error-throwing default value
        if df_det is None:
            raise ValueError("df_det must be DataFrame or 'nom'")
        ## String shortcut
        elif isinstance(df_det, str):
            if df_det == "nom":
                df_det = self.det_nom()
            else:
                raise ValueError("df_det shortcut string invalid")
        ## DataFrame
        else:
            ## Check invariant; model inputs must be subset of df columns
            if not set(self.var_det).issubset(set(df_det.columns)):
                raise ValueError("model.var_det not a subset of given columns")

        return tran_outer(df_rand, df_det)

    def var_rand_quantile(self, df_quant):
        """Convert random variable quantiles to input samples

        @param df_quant DataFrame; values \in [0,1]
        @returns DataFrame

        @pre df_quant.shape[1] == n_var_rand
        @post df_samp.shape[1] == n_var_rand
        """
        ## Check invariant; given columns must be equal to var_rand
        if (set(self.density._var_rand) != set(df_quant.columns)):
            raise ValueError("Quantile columns must equal model var_rand")

        samples = np.zeros(df_quant.shape)
        ## Ensure correct column ordering
        quantiles = df_quant[self.density._var_rand].values

        ## Perform copula conversion, if necessary
        if self.density._copula is not None:
            raise NotImplementedError
            ## Build correlation structure
            Sigma                                = np.eye(self.n_in)
            Sigma[np.triu_indices(self.n_in, 1)] = self.density.pdf_corr
            Sigma                                = Sigma + \
                                                   (Sigma - np.eye(self.n_in)).T
            Sigma_h                              = cholesky(Sigma)
            ## Convert samples
            gaussian_samples = np.dot(norm.ppf(quantiles), Sigma_h.T)
            ## Convert to uniform marginals
            quantiles = norm.cdf(gaussian_samples)
        ## Skip if no dependence structure

        ## Apply appropriate marginal
        for ind in range(len(self.density._var_rand)):
            ## Map with inverse density
            samples[:, ind] = self.density._marginals[ind].q(quantiles[:, ind])

        return pd.DataFrame(data=samples, columns=self.density._var_rand)

    def name_corr(self):
        """Name the correlation elements
        """
        raise NotImplementedError
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
        new_model = model(
            name     = self.name,
            function = self.function,
            outputs  = self.outputs,
            domain   = self.domain,
            density  = self.density
        )
        return new_model

    def printpretty(self):
        """Formatted print of model attributes
        """
        print("model: {}".format(self.name))

        print("  var_det:")
        for var_det in self.var_det:
            print("    {}".format(var_det))

        print("  var_rand:")
        for marginal in self.density._marginals:
            print("    {}".format(marginal.summary()))

        print("  outputs:")
        for output in self.outputs:
            print("    {}".format(output))

# Derived dataframe-vectorized model
class model_vectorized(model):
    """Derived class for grama models.

    Given function must be vectorized over dataframes
    """

    def evaluate_df(self, df):
        """Evaluate function using an input dataframe

        Assumes function is vectorized over dataframes.
        """
        return self.function(df)

    def copy(self):
        """Make a copy of this model
        """
        new_model = model_vectorized(
            name     = self.name,
            function = self.function,
            outputs  = self.outputs,
            domain   = self.domain,
            density  = self.density
        )
        return new_model
