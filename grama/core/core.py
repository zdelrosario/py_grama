## grama core functions
# Zachary del Rosario, March 2019

__all__ = [
    "Domain",
    "Density",
    "Function",
    "FunctionVectorized",
    "MarginalNamed",
    "Model",
    "valid_dist"
]

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from itertools import chain
from numpy.linalg import cholesky
from ..transforms import tran_outer
from ..tools import pipe
from toolz import curry

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
# Function class
class Function:
    """Parent class for functions.

    A function specifies its own inputs and outputs; these are subsets of the
    full model's inputs and outputs.

    """
    def __init__(self, func, inputs, outputs, name):
        """Constructor

        :param func: Function mapping X^d -> X^r
        :param inputs: Named inputs; must match order of X^d
        :param outputs: Named outputs; must match order of X^r
        :param name: Function name

        :type func: function
        :type inputs: List of Strings
        :type outputs: List of Strings
        :type name: String

        """
        self.func = func
        self.inputs = inputs
        self.outputs = outputs
        self.name = name

    def eval(self, df):
        """Evaluate function. Loops over dataframe rows.

        :param df: Input values to evaluate
        :type df: Pandas DataFrame

        :returns: Result values
        :rtype: Pandas DataFrame

        """
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.inputs).issubset(set(df.columns)):
            raise ValueError(
                "Model function `{}` inputs not a subset of given columns".format(self.name)
            )

        ## Set up output
        n_rows  = df.shape[0]
        results = np.zeros((n_rows, len(self.outputs)))
        for ind in range(n_rows):
            results[ind] = self.func(df.loc[ind, self.inputs])

        ## Package output as DataFrame
        return pd.DataFrame(data=results, columns=self.outputs)

    def summary(self):
        """Returns a summary string
        """
        return "{0:}: {1:} -> {2:}".format(self.name, self.inputs, self.outputs)

class FunctionVectorized(Function):
    def eval(self, df):
        """Evaluate function. Assumes function is vectorized over dataframes.

        :param df: Input values to evaluate
        :type df: Pandas DataFrame

        :returns: Result values
        :rtype: Pandas DataFrame

        """
        df_res = self.func(df)
        df_res.columns = self.outputs

        return df_res

# Domain parent class
class Domain:
    """Parent class for input domains

    The domain defines constraints on the variables. Together with a model's
    functions, it defines the mathematical domain of a model.

    """
    def __init__(self, bounds=None, feasible=None):
        """Initialize

        @param bounds [dict] Variable bounds, given as {"var": [Lower, Upper]} dict entires
        @param feasible [function] Vectorized function mapping variables to bool

        @pre isinstance(bounds, dict)
        @pre (feasible is function) | (feasible is None)
        @pre len(bounds) == n where model.function:R^n -> R^m

        """
        self.bounds    = bounds
        self.feasible  = feasible

    def bound_summary(self, bound):
        if bound in self.bounds.keys():
            return "{0:}: [{1:}, {2:}]".format(
                bound,
                self.bounds[bound][0],
                self.bounds[bound][1],
            )
        else:
            return "{0:}: (unbounded)".format(bound)

# Marginal parent class
class Marginal(ABC):
    """Parent class for marginal distributions
    """
    def __init__(self, var, sign=0):
        self.var = var
        self.sign = sign

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
    """Marginal using a named distribution from core.valid_dist"""

    def __init__(self, var, d_name=None, d_param=None, **kw):
        super().__init__(var, **kw)

        self.d_name = d_name
        self.d_param = d_param

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
        return "{0:} ({1:}): {2:}, {3:}".format(
            self.var,
            self.sign,
            self.d_name,
            self.d_param
        )

# Density parent class
class Density:
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
        self.marginals = marginals
        self.copula = copula
        self.var_rand = list(map(lambda m: m.var, self.marginals))

# Model parent class
class Model:
    """Parent class for grama models.
    """

    def __init__(
            self,
            name=None,
            functions=None,
            domain=None,
            density=None,
    ):
        """Constructor

        @param name [string] Name of model
        @param functions [list(gr.function)] Define the model mapping f(x) : R^n_in -> R^n_out
               along with function input and output names
        @param domain [gr.domain] Model domain
        @param density [gr.density] Model density

        @pre len(domain.inputs) == n_in
        @pre len(outputs) == n_out
        @pre isinstance(domain, domain_)
        @pre isinstance(density, density_) || (density is None)
        """
        if functions is None:
            functions = []
        if domain is None:
            domain = Domain()
        if density is None:
            density = Density()

        self.name      = name
        self.functions = functions
        self.domain    = domain
        self.density   = density

        self.update()

    def update(self):
        """Update model public attributes based on functions, domain, and density.

        The variables and parameters are implicitly defined by the model
        attributes.

        - self.functions defines the full list of inputs
        - self.domain defines the constraints on the model's domain
        - self.density defines the random variables

        """
        ## Compute list of outputs
        self.outputs = list(set().union(
            *[f.outputs for f in self.functions]
        ))

        ## Compute list of variables and parameters
        self.var = list(set().union(
            *[f.inputs for f in self.functions]
        ))
        self.var_rand = self.density.var_rand
        self.var_det  = list(set(self.var).difference(self.var_rand))

        ## TODO parameters

        ## Convenience constants
        self.n_var      = len(self.var)
        self.n_var_rand = len(self.var_rand)
        self.n_var_det  = len(self.var_det)
        self.n_out      = len(self.outputs)

    def det_nom(self):
        """Return nominal conditions for deterministic variables

        @returns [DataFrame] Nominal values for deterministic variables
        """
        data = {}

        for var in self.var_det:
            if var in self.domain.bounds.keys():
                data[var] = [0.5 * (
                        self.domain.bounds[var][0] + \
                        self.domain.bounds[var][1]
                )]
            else:
                data[var] = [0.]

        return pd.DataFrame(data=data)

    def evaluate_df(self, df):
        """Evaluate function using an input dataframe

        @param df [DataFrame] Variable values at which to evaluate model functions

        @returns [DataFrame] Output results
        """
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.var).issubset(set(df.columns)):
            raise ValueError("Model inputs not a subset of given columns")

        # ## Set up output
        # n_rows  = df.shape[0]
        # results = np.zeros((n_rows, self.n_out))
        # for ind in range(n_rows):
        #     results[ind] = self.function(df.loc[ind, self.domain._variables])

        ## Package output as DataFrame
        # return pd.DataFrame(data=results, columns=self.outputs)

        list_df = []
        ## Evaluate each function
        for func in self.functions:
            list_df.append(func.eval(df))

        return pd.concat(list_df, axis=1)

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
        if (set(self.density.var_rand) != set(df_quant.columns)):
            raise ValueError("Quantile columns must equal model var_rand")

        samples = np.zeros(df_quant.shape)
        ## Ensure correct column ordering
        quantiles = df_quant[self.density.var_rand].values

        ## Perform copula conversion, if necessary
        if self.density.copula is not None:
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
        for ind in range(len(self.density.var_rand)):
            ## Map with inverse density
            samples[:, ind] = self.density.marginals[ind].q(quantiles[:, ind])

        return pd.DataFrame(data=samples, columns=self.density.var_rand)

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
        new_model = Model(
            name      = self.name,
            functions = self.functions,
            domain    = self.domain,
            density   = self.density
        )
        new_model.update()

        return new_model

    def printpretty(self):
        """Formatted print of model attributes
        """
        print("model: {}".format(self.name))
        print("")
        print("  inputs:")
        print("    var_det:")
        for var_det in self.var_det:
            print("      {}".format(self.domain.bound_summary(var_det)))

        print("    var_rand:")
        for marginal in self.density._marginals:
            print("      {}".format(marginal.summary()))

        print("  functions:")
        for function in self.functions:
            print("    {}".format(function.summary()))
