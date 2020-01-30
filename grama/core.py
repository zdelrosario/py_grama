## grama core functions
# Zachary del Rosario, March 2019

__all__ = [
    "CopulaIndependence",
    "CopulaGaussian",
    "Domain",
    "Density",
    "Function",
    "FunctionVectorized",
    "MarginalNamed",
    "Model"
]

from abc import ABC, abstractmethod
import copy
# import numpy as np
# import pandas as pd
from numpy import zeros, triu_indices, eye, array
from numpy.random import random, multivariate_normal
from numpy.random import seed as set_seed
from pandas import DataFrame, concat

import grama as gr
from grama import pipe, valid_dist, param_dist

from itertools import chain
from numpy.linalg import cholesky
from toolz import curry

## Package settings
RUNTIME_LOWER = 1 # Cutoff threshold for runtime messages

## Core functions
##################################################
# Function class
class Function:
    """Parent class for functions.

    A function specifies its own inputs and outputs; these are subsets of the
    full model's inputs and outputs.

    """
    def __init__(self, func, var, out, name, runtime):
        """Function constructor

        Construct a grama function. Generally not called directly; preferred
        usage is through gr.comp_function().

        Args:
            func (function): Function mapping X^d -> X^r
            var (list(str)): Named variables; must match order of X^d
            out (list(str)): Named outputs; must match order of X^r
            name (str): Function name
            runtime (numeric): Estimated single-eval runtime (in seconds)

        Returns:
            gr.Function: grama function

        """
        self.func = func
        self.var = var
        self.out = out
        self.name = name
        self.runtime = runtime

    def copy(self):
        """Make a copy"""
        func_new = Function(
            copy.deepcopy(self.func),
            copy.deepcopy(self.var),
            copy.deepcopy(self.out),
            copy.deepcopy(self.name),
            runtime=self.runtime
        )
        return func_new

    def eval(self, df):
        """Evaluate function

        Evaluate a grama function; loops over dataframe rows. Intended for
        internal use.

        Args:
        df (DataFrame): Input values to evaluate

        Returns:
            DataFrame: Result values

        """
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.var).issubset(set(df.columns)):
            raise ValueError(
                "Model function `{}` var not a subset of given columns".format(self.name)
            )

        ## Set up output
        n_rows  = df.shape[0]
        results = zeros((n_rows, len(self.out)))
        for ind in range(n_rows):
            results[ind] = self.func(df.loc[ind, self.var])

        ## Package output as DataFrame
        return DataFrame(data=results, columns=self.out)

    def summary(self):
        """Returns a summary string
        """
        return "{0:}: {1:} -> {2:}".format(self.name, self.var, self.out)

class FunctionVectorized(Function):
    def eval(self, df):
        """Evaluate function; DataFrame vectorized

        Evaluate grama function. Assumes function is vectorized over dataframes.

        Args:
            df (DataFrame): Input values to evaluate

        Returns:
            DataFrame: Result values

        """
        df_res = self.func(df)
        df_res.columns = self.out

        return df_res

    def copy(self):
        """Make a copy"""
        func_new = FunctionVectorized(
            self.func,
            self.var,
            self.out,
            self.name,
            self.runtime
        )
        return func_new

# Domain parent class
class Domain:
    """Parent class for input domains

    The domain defines constraints on the variables. Together with a model's
    functions, it defines the mathematical domain of a model.

    """
    def __init__(self, bounds=None, feasible=None):
        """Constructor

        Construct a grama domain. Generally not called directly; preferred usage
        is through gr.comp_bounds().

        Args:
        bounds (dict): Variable bounds, given as {"var": [Lower, Upper]}
            dict entires
        feasible (function): Vectorized function mapping variables to bool
            NOT IMPLEMENTED

        Returns:
            gr.Domain: grama domain

        @pre isinstance(bounds, dict)
        @pre (feasible is function) | (feasible is None)
        @pre len(bounds) == n where model.function:R^n -> R^m

        """
        if bounds is None:
            bounds = {}

        self.bounds   = bounds
        self.feasible = feasible
        self.var = bounds.keys()

    def copy(self):
        new_domain = Domain(
            bounds=copy.deepcopy(self.bounds),
            feasible=copy.deepcopy(self.feasible)
        )

        return new_domain

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
    def __init__(self, sign=0):
        self.sign = sign

    @abstractmethod
    def copy(self):
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
            sign=self.sign,
            d_name=self.d_name,
            d_param=copy.deepcopy(self.d_param)
        )

        return new_marginal

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

## Copula base class
class Copula(ABC):
    """Parent class for copulas
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def sample(self, n=1):
        pass

    @abstractmethod
    def summary(self):
        pass

class CopulaIndependence(Copula):
    def __init__(self, var_rand):
        """Constructor

        Args:
            var_rand (ind): Number of random variables

        Returns:
            gr.CopulaIndependence: Independence copula

        """
        self.var_rand = var_rand

    def copy(self):
        """Copy

        Returns:
            gr.CopulaIndependence: Copy of present copula

        """
        cop = CopulaIndependence(var_rand=self.var_rand)

        return cop

    def sample(self, n=1, seed=None):
        """Draw samples from copula

        Args:
            n (int): Number of samples
            seed (int): Random seed

        Returns:
            DataFrame: Independent samples
        """
        ## Set seed only if given
        if seed is not None:
            set_seed(seed)

        return DataFrame(
            data=random((n, len(self.var_rand))),
            columns=self.var_rand
        )

    def summary(self):
        return "Independence copula"

class CopulaGaussian(Copula):
    def __init__(self, df_corr):
        """Constructor

        Args:
            self (gr.CopulaGaussian):
            df_corr (DataFrame): Pairwise correlations provided as
                off-diagonal upper-triangular entries. Must have columns
                ["var1", "var2", "corr"]. Zero-correlation entires must
                be provided.

        Returns:
            gr.CopulaGaussian: Gaussian copula

        """
        ## Assemble data
        df_corr.sort_values(["var1", "var2"], inplace=True)
        df_summary = df_corr.set_index(["var1", "var2"]) \
                            .count(level="var1") \
                            .sort_values("corr", ascending=False)
        var_rand = list(df_summary.index) + [df_corr.var2.iloc[-1]]
        n_var_rand = df_summary.iloc[0, 0] + 1
        Ind_upper = triu_indices(n_var_rand, 1)
        ## Check invariants
        if len(var_rand) != n_var_rand:
            raise ValueError("Invalid set of correlations provided")
        if len(Ind_upper[0]) != len(df_corr["corr"]):
            raise ValueError("Invalid set of correlations provided")

        ## Build correlation structure
        Sigma            = eye(n_var_rand)
        Sigma[Ind_upper] = df_corr["corr"].values
        Sigma            = Sigma + \
                           (Sigma - eye(n_var_rand)).T
        Sigma_h          = cholesky(Sigma)

        self.df_corr = df_corr
        self.var_rand = var_rand
        self.Sigma   = Sigma
        self.Sigma_h = Sigma_h

    def copy(self):
        """Copy

        Args:
            self (gr.CopulaGaussian):

        Returns:
            gr.CopulaGaussian:
        """
        cop = CopulaGaussian(df_corr=self.df_corr.copy())

        return cop

    def sample(self, n=1, seed=None):
        """Draw samples from copula

        Draw samples according to gaussian copula dependence structure.

        Args:
            self (gr.CopulaGaussian):
            n (int): Number of samples to draw

        Returns:
            array: Copula samples

        """
        ## Set seed only if given
        if seed is not None:
            set_seed(seed)

        ## Generate correlated samples
        gaussian_samples = multivariate_normal(
            mean=[0] * len(self.var_rand),
            cov=self.Sigma,
            size=n
        )
        ## Convert to uniform marginals
        quantiles = valid_dist["norm"].cdf(gaussian_samples)

        return DataFrame(
            data=quantiles,
            columns=self.var_rand
        )

    def summary(self):
        return "Gaussian copula with correlations:\n{}".format(self.df_corr)

## Density parent class
class Density:
    """Parent class for joint densities

    The density is defined for all the random variables; therefore it explicitly
    defines the list of random variables, and together implicitly defines the
    deterministic variables via

        domain.var + [functions.var] - density.marginals.keys()

    """
    def __init__(self, marginals=None, copula=None):
        """Constructor

        Construct a grama density. Generally not called directly; preferred
        usage is through gr.comp_marginals() and gr.comp_copula().

        Args:
            marginals (dict): Dictionary of gr.Marginal objects
            copula (gr.Copula): Copula object

        Returns:
            gr.Density: grama density

        """
        self.marginals = marginals
        self.copula = copula

    def copy(self):
        try:
            new_marginals = {}
            for key, value in self.marginals.items():
                new_marginals[key] = self.marginals[key].copy()

        except AttributeError:
            new_marginals = {}

        try:
            new_copula = self.copula.copy()
        except AttributeError:
            new_copula = None

        new_density = Density(
            marginals=new_marginals,
            copula=new_copula
        )

        return new_density

    def pr2sample(self, df_prval):
        """Convert CDF probabilities to samples

        Convert random variable CDF probabilities to random variable samples.
        Ignores dependence structure.

        Args:
            df_prval (DataFrame): Values \in [0,1]

        Returns:
            DataFrame: Variable samples (quantiles)

        @pre df_prval.shape[1] == len(self.var_rand)
        @post result.shape[1] == len(self.var_rand)

        """
        try:
            var_rand = list(self.marginals.keys())
        except AttributeError:
            var_rand = []

        ## Empty case
        if len(var_rand) == 0:
            return DataFrame()

        ## Variables to convert
        var_comp = list(set(var_rand).intersection(set(df_prval.columns)))
        if len(var_comp) == 0:
            raise ValueError(
                "Intersection of df_prval.columns and var_rand must be nonempty"
            )

        samples = zeros(df_prval[var_comp].shape)
        ## Ensure correct column ordering
        prval = df_prval[var_comp].values

        ## Apply appropriate marginal
        for ind in range(len(var_comp)):
            ## Map with inverse density
            var = var_comp[ind]
            samples[:, ind] = self.marginals[var].q(prval[:, ind])

        return DataFrame(data=samples, columns=var_comp)

    def sample2pr(self, df_sample):
        """Convert samples to CDF probabilities

        Convert random variable samples to CDF probabilities. Ignores dependence
        structure.

        Args:
            df_sample (DataFrame): Values \in [0,1]

        Returns:
            DataFrame: Variable samples (quantiles)

        @pre df_sample.shape[1] == len(self.var_rand)
        @post result.shape[1] == len(self.var_rand)

        """
        try:
            var_rand = list(self.marginals.keys())
        except AttributeError:
            var_rand = []

        ## Empty case
        if len(var_rand) == 0:
            return DataFrame()

        ## Variables to convert
        var_comp = list(set(var_rand).intersection(set(df_sample.columns)))
        if len(var_comp) == 0:
            raise ValueError(
                "Intersection of df_sample.columns and var_rand must be nonempty"
            )

        prval = zeros(df_sample[var_comp].shape)
        ## Ensure correct column ordering
        sample = df_sample[var_comp].values

        ## Apply appropriate marginal
        for ind in range(len(var_comp)):
            ## Map with inverse density
            var = var_comp[ind]
            prval[:, ind] = self.marginals[var].p(sample[:, ind])

        return DataFrame(data=prval, columns=var_comp)

    def sample(self, n=1, seed=None):
        """Draw samples from joint density

        Draw samples according to joint density using marginal and copula
        information.

        Args:
            n (int): Number of samples to draw
            seed (int): random seed to use

        Returns:
            DataFrame: Joint density samples

        """
        if not (self.copula is None):
            df_pr = self.copula.sample(n=n, seed=seed)
        else:
            raise ValueError(
                "\n" + \
                "Present model copula must be defined for sampling.\n" + \
                "Use CopulaIndependence only when inputs can be guaranteed\n" + \
                "independent. See the Documentation chapter on Random\n" + \
                "Variable Modeling for more information."
            )
        return self.pr2sample(df_pr)

    def summary_marginal(self, var):
        return "{0:}: {1:}".format(var, self.marginals[var].summary())

    def summary_copula(self):
        if not (self.copula is None):
            return self.copula.summary()
        else:
            return None

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

        Construct a grama model. Generally called without arguments; suggested
        procedure is to use gr.building tools to build up the model.

        Args:
            name (string): Name of model
            functions (list(gr.function)): Define the model mapping
                f(x) : R^n_in -> R^n_out along with function input and output names
            domain (gr.Domain): Model domain
            density (gr.Density): Model density

        Returns:
            gr.Model: grama model

        @pre len(domain.var) == n_in
        @pre len(out) == n_out
        @pre isinstance(domain, domain_)
        @pre isinstance(density, density_) || (density is None)

        Examples:

            >>> import grama as gr
            >>> print(gr.valid_dist.keys()) # Supported distributions
            >>> md = gr.Model() >> \
            >>>     gr.cp_function(
            >>>         lambda x: x[0] + x[1],
            >>>         var=["x0", "x1"],
            >>>         out=1
            >>>     ) >> \
            >>>     gr.cp_marginals(
            >>>         x0={"dist": "uniform", "loc": 0, "scale": 1}
            >>>     ) >> \
            >>>     gr.cp_bounds(x1=(0, 1))

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
        attributes. For internal use.

        - self.functions defines the full list of inputs
        - self.domain defines the constraints on the model's domain
        - self.density defines the random variables

        """
        ## Compute list of outputs
        self.out = list(set().union(
            *[f.out for f in self.functions]
        ))

        ## Compute list of variables and parameters
        self.var = list(set().union(
            *[f.var for f in self.functions]
        ).union(set(self.domain.var)))

        try:
            self.var_rand = list(self.density.marginals.keys())
        except AttributeError:
            self.var_rand = []
        self.var_det  = list(set(self.var).difference(self.var_rand))

        ## TODO parameters

        ## Convenience constants
        self.n_var      = len(self.var)
        self.n_var_rand = len(self.var_rand)
        self.n_var_det  = len(self.var_det)
        self.n_out      = len(self.out)

    def runtime(self, n):
        """Estimate runtime

        Estimate the total runtime to evaluate n observations.

        Args:
            self (gr.Model):
            n (int): Number of observations

        Returns:
            float: Estimated runtime, in seconds

        """
        rate = 0
        for fun in self.functions:
            rate = rate + fun.runtime

        return rate * n

    def runtime_message(self, df):
        """Runtime message

        Estimate total runtime based on proposed DataFrame, prepare a message
        for console print. Print no message if runtime is negligible.

        Args:
            self (gr.Model):
            df (DataFrame): Data to evaluate

        Returns:
            str: Runtime message

        """
        runtime = self.runtime(df.shape[0])

        if runtime < RUNTIME_LOWER:
            return None
        else:
            return "Estimated runtime: {0:3.4f} sec".format(runtime)

    def det_nom(self):
        """Return nominal conditions for deterministic variables

        Returns:
            DataFrame: Nominal values for deterministic variables

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

        return DataFrame(data=data)

    def evaluate_df(self, df):
        """Evaluate function using an input dataframe

        Args:
            df (DataFrame): Variable values at which to evaluate model functions

        Returns:
            DataFrame: Output results

        """
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.var).issubset(set(df.columns)):
            raise ValueError("Model inputs not a subset of given columns")

        list_df = []
        ## Evaluate each function
        for func in self.functions:
            list_df.append(func.eval(df))

        return concat(list_df, axis=1)

    def var_outer(self, df_rand, df_det=None):
        """Outer product of random and deterministic samples

        Args:
            df_rand (DataFrame) random variable samples
            df_det (DataFrame) deterministic variable samples
                set to "nom" for nominal evaluation

        Returns:
            DataFrame: Outer product of samples

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

        ## Pass-through if no var_rand
        if self.n_var_rand == 0:
            return df_det

        ## Outer product if both det and rand exist
        return gr.tran_outer(df_rand, df_det)

    def name_corr(self):
        """Name the correlation elements
        """
        raise NotImplementedError
        ## Build matrix of names
        corr_mat = []
        for ind in range(self.n_in):
            corr_mat.append(
                list(map(
                    lambda s: s + "," + self.domain.var[ind],
                    self.domain.var
                ))
            )

        ## Access matrix of names
        corr_names = dict()
        corr_ind = triu_indices(self.n_in, 1)
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
            functions = copy.deepcopy(self.functions),
            domain    = self.domain.copy(),
            density   = self.density.copy()
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
        try:
            for key, marginal in self.density.marginals.items():
                print("      {}".format(self.density.summary_marginal(key)))
        except AttributeError:
            pass
        print("    copula:")
        print("        {}".format(self.density.summary_copula()))

        print("  functions:")
        for function in self.functions:
            print("    {}".format(function.summary()))
