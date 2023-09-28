## grama core functions
# Zachary del Rosario, March 2019

__all__ = [
    "CopulaIndependence",
    "CopulaGaussian",
    "Domain",
    "Density",
    "Function",
    "FunctionModel",
    "FunctionVectorized",
    "Model",
    "NaN",
]

import copy
import networkx as nx
import warnings
from grama import pipe, valid_dist, param_dist
from .tools import tran_outer
from abc import ABC, abstractmethod
from itertools import chain
from numpy import (
    ones,
    zeros,
    triu_indices,
    eye,
    array,
    Inf,
    NaN,
    sqrt,
    dot,
    diag,
    isfinite,
    prod,
    exp,
)
from numpy import min as npmin
from numpy import max as npmax
from numpy.linalg import cholesky, det, inv
from numpy.random import random, multivariate_normal, rand
from numpy.random import seed as set_seed
from pandas import DataFrame, concat
from scipy.linalg import det, LinAlgError, solve
from scipy.optimize import root_scalar
from scipy.stats import norm, gaussian_kde
from toolz import curry


## Package settings
RUNTIME_LOWER = 1  # Cutoff threshold for runtime messages

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
            runtime=self.runtime,
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
                "Model function `{}` var not a subset of given columns".format(
                    self.name
                )
            )

        ## Set up output
        n_rows = df.shape[0]
        results = zeros((n_rows, len(self.out)))

        for ind in range(n_rows):
            results[ind] = self.func(*df.loc[ind, self.var])

        ## Package output as DataFrame
        return DataFrame(data=results, columns=self.out)

    def summary(self):
        """Returns a summary string"""
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
        return df_res[self.out]

    def copy(self):
        """Make a copy"""
        func_new = FunctionVectorized(
            self.func, self.var, self.out, self.name, self.runtime
        )
        return func_new


class FunctionModel(Function):
    """gr.Model as gr.Function"""

    def __init__(self, md, ev=None, var=None, out=None):
        """Model-Function constructor

        Construct a grama function from a model. Generally not called directly;
        preferred usage is through gr.comp_model().

        Args:
            md (gr.Model): Grama model
            ev (function): Evaluation function for model; must have signature
                ev(md, df); must take df with columns matching given `var`
            var (list): Variables used by ev() to evaluate md. Ignored if
                default ev used.
            out (list): Outputs returned by ev(). Ignored if default ev used.

        Returns:
            gr.Function: grama function

        """
        self.model = md

        ## Construct default evaluator
        if ev is None:

            def _ev(md, df):
                df_res = md.evaluate_df(df)
                return df_res[md.out]

            self.ev = _ev
            self.var = self.model.var
            self.out = self.model.out

        ## Use given evaluator
        else:
            self.ev = ev
            self.var = var
            self.out = out

        ## Copy model data
        self.runtime = md.runtime(1)
        self.name = copy.copy(md.name)

    def eval(self, df):
        """Evaluate function; DataFrame vectorized

        Evaluate grama model as a function. Modify the parameters before

        Args:
            df (DataFrame): Input values to evaluate

        Returns:
            DataFrame: Result values

        """
        return self.ev(self.model, df)

    def copy(self):
        """Make a copy"""
        func_new = FunctionModel(self.model, ev=self.ev, var=self.var, out=self.out)
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

        self.bounds = bounds
        self.feasible = feasible
        self.update()

    def update(self):
        self.var = list(self.bounds.keys())

    def copy(self):
        new_domain = Domain(
            bounds=copy.deepcopy(self.bounds), feasible=copy.deepcopy(self.feasible)
        )

        return new_domain

    def get_bound(self, var):
        if var in self.bounds.keys():
            return (self.bounds[var][0], self.bounds[var][1])
        return (-Inf, +Inf)

    def get_width(self, var):
        if var in self.bounds.keys():
            return self.bounds[var][1] - self.bounds[var][0]
        return +Inf

    def get_nominal(self, var):
        if var in self.bounds.keys():
            if isfinite(self.bounds[var][0]):
                if isfinite(self.bounds[var][1]):
                    return 0.5 * (self.bounds[var][1] + self.bounds[var][0])
                return self.bounds[var][0]
            if isfinite(self.bounds[var][1]):
                return self.bounds[var][1]
            return NaN
        return NaN

    def bound_summary(self, var):
        if var in self.bounds.keys():
            return "{0:}: [{1:}, {2:}]".format(
                var,
                self.bounds[var][0],
                self.bounds[var][1],
            )
        return "{0:}: (unbounded)".format(var)


## Copula base class
class Copula(ABC):
    """Parent class for copulas"""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, n=1):
        r"""Draw a sample of a given size"""
        raise NotImplementedError

    @abstractmethod
    def d(self, u):
        r"""Copula density"""
        raise NotImplementedError

    @abstractmethod
    def u2z(self, u):
        r"""Transform from [0, 1]^d to sample space"""
        raise NotImplementedError

    @abstractmethod
    def z2u(self, z):
        r"""Transform from sample space to [0, 1]^d"""
        raise NotImplementedError

    @abstractmethod
    def dudz(self, z):
        r"""Jacobian of copula transform"""
        raise NotImplementedError

    @abstractmethod
    def summary(self):
        raise NotImplementedError


class CopulaIndependence(Copula):
    def __init__(self, var_rand, source="real"):
        """Constructor

        Args:
            var_rand (ind): Number of random variables

        Returns:
            gr.CopulaIndependence: Independence copula

        """
        self.var_rand = var_rand
        if source not in ["real", "error"]:
            raise ValueError(
                "Your source of variability must be either 'real' or 'error'!"
            )
        self.source = source

    def copy(self):
        """Copy

        Returns:
            gr.CopulaIndependence: Copy of present copula

        """
        cop = CopulaIndependence(var_rand=self.var_rand, source=self.source)

        return cop

    def sample(self, n=1, seed=None, var_name=None):
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

        if var_name is not None:
            return DataFrame(data=random((n)), columns=var_name)
        else:
            return DataFrame(
                data=random((n, len(self.var_rand))), columns=self.var_rand
            )

    def d(self, u):
        """Density function

        Args:
            u (array-like):

        Returns:
            array: Copula density values

        """
        return ones(u.shape[0])

    def u2z(self, u):
        """Transform to standard-normal space

        Args:
            u (array-like):

        Returns:
            array:

        """
        return norm.ppf(u)

    def z2u(self, z):
        """Transform to uniform-marginal space

        Args:
            z (array-like):

        Returns:
            array:

        """
        return norm.cdf(z)

    def dudz(self, z):
        """Jacobian

        Args:
            z (array-like)

        Returns:
            array:

        """
        return diag(norm.pdf(z))

    def summary(self):
        return f"Independence copula (source: {self.source})"


class CopulaGaussian(Copula):
    def __init__(self, var_rand, df_corr, source="real"):
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
        ## Sort correlations along given var_rand
        sorter_index = dict(zip(var_rand, range(len(var_rand))))
        df_corr["var1_rank"] = df_corr["var1"].map(sorter_index)
        df_corr["var2_rank"] = df_corr["var2"].map(sorter_index)
        df_corr.sort_values(
            ["var1_rank", "var2_rank"], ascending=[True, True], inplace=True
        )
        df_corr.drop(["var1_rank", "var2_rank"], axis=1, inplace=True)

        n_var_rand = len(var_rand)
        Ind_upper = triu_indices(n_var_rand, 1)

        ## Check invariants
        if len(Ind_upper[0]) != len(df_corr["corr"]):
            raise ValueError("Invalid set of correlations provided")

        ## Build correlation structure
        Sigma = eye(n_var_rand)
        Sigma[Ind_upper] = df_corr["corr"].values
        Sigma = Sigma + (Sigma - eye(n_var_rand)).T
        try:
            Sigma_h = cholesky(Sigma)
            Sigma_i = inv(Sigma)
        except LinAlgError:
            warnings.warn(
                "Correlation structure is not positive-definite; copula transforms not available",
                RuntimeWarning,
            )
            Sigma_h = None
            Sigma_i = None

        ## Build density quantities

        self.df_corr = df_corr
        self.var_rand = var_rand
        if source not in ["real", "error"]:
            raise ValueError(
                "Your source of variability must be either 'real' or 'error'!"
            )
        self.source = source
        self.Sigma = Sigma
        self.Sigma_h = Sigma_h
        self.Sigma_i = Sigma_i
        self.det = det(Sigma)

    def copy(self):
        """Copy

        Args:
            self (gr.CopulaGaussian):

        Returns:
            gr.CopulaGaussian:
        """
        cop = CopulaGaussian(self.var_rand, self.df_corr.copy(), self.source)
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
            mean=[0] * len(self.var_rand), cov=self.Sigma, size=n
        )
        ## Convert to uniform marginals
        quantiles = valid_dist["norm"].cdf(gaussian_samples)

        return DataFrame(data=quantiles, columns=self.var_rand)

    def d(self, u):
        """Copula density function

        Args:
            u (array-like):

        Returns:
            array:

        """
        n_obs = u.shape[0]
        n_dim = u.shape[1]
        l_values = zeros(n_obs)

        for i in range(n_obs):
            v = norm.ppf(u[i, :])
            l_values[i] = exp(-0.5 * dot(v, dot(self.Sigma_i - eye(n_dim), v))) / sqrt(
                self.det
            )

        return l_values

    def u2z(self, u):
        """Transform to standard-normal space

        Args:
            u (array-like):

        Returns:
            array:

        """
        N = norm.ppf(u)
        Z = solve(self.Sigma_h, N)

        return Z

    def z2u(self, z):
        """Transform to uniform-marginal space

        Args:
            z (array-like):

        Returns:
            array:

        """
        return norm.cdf(dot(self.Sigma_h, z))

    def dudz(self, z):
        """Jacobian

        Args:
            z (array-like):

        Returns:
            array:

        """
        return dot(self.Sigma_h.T, diag(norm.pdf(dot(self.Sigma_h, z))))

    def summary(self):
        return f"Gaussian copula (source: {self.source}) with correlations:\n{self.df_corr}"


## Density parent class
class Density:
    """Parent class for joint densities

    The density is defined for all the random variables; therefore it explicitly
    defines the list of random variables, and together implicitly defines the
    deterministic variables via

        domain.var + [functions.var] - density.marginals.keys()

    """

    def __init__(self, marginals=None, copula_real=None, copula_err=None):
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
        self.copula_err = copula_err
        self.copula_real = copula_real

    def copy(self):
        try:
            new_marginals = {}
            for key, value in self.marginals.items():
                new_marginals[key] = self.marginals[key].copy()

        except AttributeError:
            new_marginals = {}

        try:
            new_copula_err = self.copula_err.copy()
        except AttributeError:
            new_copula_err = None

        try:
            new_copula_real = self.copula_real.copy()
        except AttributeError:
            new_copula_real = None

        new_density = Density(
            marginals=new_marginals,
            copula_err=new_copula_err,
            copula_real=new_copula_real,
        )

        return new_density

    def check_valid_sources(self):
        """
        Checks that types of copula within the density match the sources of the marginals.
        """
        copula_sources = []
        copula_vars = []
        if self.copula_real is not None:
            copula_sources.append("real")
            copula_vars = copula_vars + self.copula_real.var_rand
        if self.copula_err is not None:
            copula_sources.append("error")
            copula_vars = copula_vars + self.copula_err.var_rand

        marginal_sources = []
        for var in self.marginals.keys():
            marginal_sources.append(self.marginals[var].source)
            if self.copula_err is not None and self.copula_real is not None:
                if var in self.copula_real.var_rand and var in self.copula_err.var_rand:
                    raise ValueError(
                        f"Random variable {var} belongs to mulitple copulas."
                    )
            if var not in copula_vars:
                raise ValueError(
                    f"Random variable {var} does not belong to any copula."
                )
            if self.marginals[var].source not in copula_sources:
                raise ValueError(
                    f"Copula with a source type of '{self.marginals[var].source}' does not exist."
                )

        for source in copula_sources:
            if source not in marginal_sources:
                raise ValueError(
                    f"Copula of source type {source} has no matching variables."
                )

        # old checking method below
        marginal_sources = set(marginal_sources)

        if sorted(copula_sources) != sorted(marginal_sources):
            raise ValueError(
                "Sources of variability within copulas do not match sources within marginal values."
            )

    def d(self, df):
        r"""Evaluate PDF

        Evaluate the PDF of the density.

        Args:
            df (DataFrame): Values

        """
        # Get variable names
        var = []
        var_real = []
        var_err = []
        for key in self.marginals.keys():
            var.append(key)
            if self.marginals[var].source == "real":
                var_real.append(key)
            else:
                var_err.append(key)

        # Evaluate copula density
        if self.copula_real is None:
            l_copula_real = 1
        else:
            df_u_real = self.sample2pr(df)[var_real]
            l_copula_real = self.copula_real.d(df_u_real.values)

        if self.copula_err is None:
            l_copula_err = 1
        else:
            df_u_err = self.sample2pr(df)[var_err]
            l_copula_err = self.copula_err.d(df_u_err.values)

        # Evaluate marginal densities
        L_marginals = zeros((df.shape[0], len(var)))
        for i, v in enumerate(var):
            L_marginals[:, i] = self.marginals[v].d(df[v])
        l_marginals = prod(L_marginals, axis=1)

        # return l_copula * l_marginals
        return l_copula_real * l_copula_err * l_marginals

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

    def sample(self, n=None, n_r=None, n_e=None, seed=None, source_type="real"):
        """Draw samples from joint density

        Draw samples according to joint density using marginal and copula
        information.

        Args:
            n (int): Number of samples to draw
            seed (int): random seed to use

        Returns:
            DataFrame: Joint density samples

        """
        self.check_valid_sources()
        if self.copula_real is None and self.copula_err is None:
            raise ValueError(
                "\n"
                + "Present model copula must be defined for sampling.\n"
                + "Use CopulaIndependence only when inputs can be guaranteed\n"
                + "independent. See the Documentation chapter on Random\n"
                + "Variable Modeling for more information.\n"
                + "https://py-grama.readthedocs.io/en/latest/source/rv_modeling.html"
            )

        # check for different source cases
        if source_type == "real":
            df_pr = self.copula_real.sample(n=n_r, seed=seed)
        elif source_type == "error":
            df_pr = self.copula_err.sample(n=n_e, seed=seed)
        elif source_type == "mixed":
            df_real = self.copula_real.sample(n=n_r, seed=seed)
            df_err = self.copula_err.sample(n=n_e, seed=seed)
            df_pr = tran_outer(df_real, df_err)
        elif source_type == "mixed_standard":
            df_real = self.copula_real.sample(n=n, seed=seed)
            df_err = self.copula_err.sample(n=n, seed=seed)
            df_pr = concat([df_real, df_err], axis=1)
        else:
            raise ValueError(
                "Invalid source_type argument. Source type may only be 'real', 'error', 'mixed', or 'mixed_standard'."
            )

        return self.pr2sample(df_pr)

    def summary_marginal(self, var):
        return "{0:}: {1:}".format(var, self.marginals[var].summary())

    def summary_copula(self):
        summary = ""
        if self.copula_real is not None:
            summary += f"{self.copula_real.summary()}\n      "
        if self.copula_err is not None:
            summary += f"{self.copula_err.summary()}"
        return summary


# Model parent class
class Model:
    """Parent class for grama models."""

    def __init__(
        self,
        name=None,
        functions=None,
        domain=None,
        density=None,
    ):
        r"""Constructor

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
        if name is None:
            name = "(no name)"

        self.name = name
        self.functions = functions
        self.domain = domain
        self.density = density

        self.update()

    def update(self):
        """Update model public attributes based on functions, domain, and density.

        The variables and parameters are implicitly defined by the model
        attributes. For internal use.

        - self.functions defines the full list of inputs
        - self.domain defines the constraints on the model's domain
        - self.density defines the random variables

        """
        ## Initialize
        self.domain.update()
        self.var = self.domain.var.copy()
        self.out = []

        ## Construct var and out, respecting DAG properties
        for fun in self.functions:
            self.var = list(set(self.var).union(set(fun.var).difference(set(self.out))))

            self.out = list(set(self.out).union(set(fun.out)))

        try:
            self.var_rand = list(self.density.marginals.keys())
        except AttributeError:
            self.var_rand = []
        self.var_det = list(set(self.var).difference(self.var_rand))

        self.source_list = []
        self.var_rand_real = []
        self.var_rand_err = []
        for key_ind in range(0, len(self.var_rand)):
            var_key = list(self.var_rand)[key_ind]
            self.source_list.append(self.density.marginals[var_key].source)
            if self.density.marginals[var_key].source == "real":
                self.var_rand_real.append(var_key)
            else:
                self.var_rand_err.append(var_key)

        ## TODO parameters

        ## Convenience constants
        self.n_var = len(self.var)
        self.n_var_rand = len(self.var_rand)
        self.n_var_det = len(self.var_det)
        self.n_out = len(self.out)

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

        return float(rate * n)

    def runtime_message(self, df):
        """Runtime message

        Estimate total runtime based on proposed DataFrame, prepare a message
        for console print.

        Args:
            self (gr.Model):
            df (DataFrame): Data to evaluate

        Returns:
            str: Runtime message

        """
        runtime = self.runtime(df.shape[0])

        if runtime is None:
            return "Runtime estimate unavailable; model has no timing data."
        elif runtime >= 3600:
            return "Estimated runtime: {0:3.4f} hr".format(runtime / 3600)
        elif runtime >= 60:
            return "Estimated runtime: {0:3.4f} min".format(runtime / 60)
        else:
            return "Estimated runtime: {0:3.4f} sec".format(runtime)

    def det_nom(self):
        """Return nominal conditions for deterministic variables

        Returns:
            DataFrame: Nominal values for deterministic variables

        """
        data = {}

        for var in self.var_det:
            data[var] = [self.domain.get_nominal(var)]

        return DataFrame(data=data)

    def evaluate_df(self, df):
        """Evaluate function using an input dataframe

        Args:
            df (DataFrame): Variable values at which to evaluate model functions

        Returns:
            DataFrame: Output results

        """
        ## Check invariant; model inputs must be subset of df columns
        # var_diff = set(self.var).difference(set(df.columns))
        # if len(var_diff) != 0:
        #     raise ValueError(
        #         "Model inputs not a subset of given columns;\n"
        #         + "missing var = {}".format(var_diff)
        #     )

        df_tmp = df.copy().drop(self.out, axis=1, errors="ignore")
        ## Evaluate each function
        for func in self.functions:
            ## Concatenate to make intermediate results available
            df_tmp = concat((df_tmp, func.eval(df_tmp)), axis=1)

        return df_tmp[self.out]

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

        elif isinstance(df_det, str) and df_det == "nom":
            df_det = self.det_nom()
        ## DataFrame
        else:
            ## Check invariant; model inputs must be subset of df columns
            if not set(self.var_det).issubset(set(df_det.columns)):
                raise ValueError("model.var_det not a subset of given columns")

        ## Pass-through if no var_rand
        if self.n_var_rand == 0:
            return df_det

        ## Outer product if both det and rand exist
        return tran_outer(df_rand, df_det)

    ## Sample transforms
    # --------------------------------------------------
    def x2z(self, x):
        r"""Transform to standard normal space

        Transform a single vector of random variable values to standard normal
        space.

        Args:
            x (array): Single vector of values in var_rand. Order of entries
                must match self.var_rand

        Returns:
            array: Single vector of values transformed to standard normal space

        """
        ## Transform to uniform
        u = zeros(self.n_var_rand)
        for i in range(self.n_var_rand):
            u[i] = self.density.marginals[self.var_rand[i]].p(x[i])
        ## Transform to standard normal
        z = self.density.copula.u2z(u)

        return z

    def z2x(self, z):
        r"""Transform to random variable space

        Transform a single vector of normal values to the model's random
        variable space.

        Args:
            z (array): Single vector of standard normal values. Order of entries
                must match self.var_rand

        Returns:
            array: Single vector of values transformed to model random variable space

        """
        ## Correlate and map to uniform
        u = self.density.copula.z2u(z)
        ## Transform per marginal
        x = zeros(self.n_var_rand)
        for i in range(self.n_var_rand):
            x[i] = self.density.marginals[self.var_rand[i]].q(u[i])

        return x

    def dxdz(self, z):
        r"""Inverse transform jacobian

        Compute jacobian of the inverse transform X = phi^{-1}(Z)

        Args:
            z (array): Single vector of standard normal values. Order of entries
                must match self.var_rand

        Returns:
            2d array: Jacobian of inverse transform

        """
        ## Setup
        dudz = self.density.copula.dudz(z)

        x = self.z2x(z)
        F = zeros(self.n_var_rand)
        for i in range(self.n_var_rand):
            F[i] = 1 / self.density.marginals[self.var_rand[i]].d(x[i])

        return dot(dudz, diag(F))

    ## Sample transforms; DataFrame
    # --------------------------------------------------
    def rand2norm(self, df):
        r"""Transform random samples to standard normal space

        Transform a DataFrame of random variable samples to standard normal space

        Args:
            df (DataFrame): Random variable samples; must have columns for all
                of self.var_rand.

        Returns:
            DataFrame: Samples in standard-normal space
        """
        ## Check invariants
        if not set(self.var_rand).issubset(set(df.columns)):
            raise ValueError("model.var_rand must be subset of df.columns")

        data = zeros((df.shape[0], self.n_var_rand))
        for i in range(df.shape[0]):
            data[i] = self.x2z(df[self.var_rand].iloc[i].values)

        return DataFrame(data=data, columns=self.var_rand)

    def norm2rand(self, df):
        r"""Transform standard normal samples to model random variable space

        Transform a DataFrame of standard normal samples to model random
        variable space.

        Args:
            df (DataFrame): Random variable samples; must have columns for all
                of self.var_rand.

        Returns:
            DataFrame: Samples in standard-normal space

        """
        ## Check invariants
        if not set(self.var_rand).issubset(set(df.columns)):
            raise ValueError("model.var_rand must be subset of df.columns")

        data = zeros((df.shape[0], self.n_var_rand))
        for i in range(df.shape[0]):
            data[i] = self.z2x(df[self.var_rand].iloc[i].values)

        return DataFrame(data=data, columns=self.var_rand)

    def name_corr(self):
        """Name the correlation elements"""
        raise NotImplementedError
        ## Build matrix of names
        corr_mat = []
        for ind in range(self.n_in):
            corr_mat.append(
                list(map(lambda s: s + "," + self.domain.var[ind], self.domain.var))
            )

        ## Access matrix of names
        corr_names = dict()
        corr_ind = triu_indices(self.n_in, 1)
        for knd in range(len(corr_ind[0])):
            ind = corr_ind[0][knd]
            jnd = corr_ind[1][knd]
            corr_names["corr_" + str(knd)] = corr_mat[ind][jnd]

        return corr_names

    ## Infrastructure
    # -------------------------
    def copy(self):
        """Make a copy of this model"""
        new_model = Model(
            name=self.name,
            functions=copy.deepcopy(self.functions),
            domain=self.domain.copy(),
            density=self.density.copy(),
        )
        new_model.update()

        return new_model

    ## Model information
    # -------------------------
    def string_rep(self):
        l = [
            "model: {}".format(self.name),
            "",
            "  inputs:",
            "    var_det:",
            "".join(
                [
                    "      {}\n".format(self.domain.bound_summary(var_det))
                    for var_det in self.var_det
                ]
            ),
            "    var_rand:",
        ]

        try:
            l = l + [
                "".join(
                    [
                        "      {}\n".format(self.density.summary_marginal(var_rand))
                        for var_rand in self.density.marginals.keys()
                    ]
                ),
            ]
        except AttributeError:
            l = l + ["\n"]

        l = l + [
            "    copula:",
            "      {}\n".format(self.density.summary_copula()),
            "  functions:",
            "".join(
                ["      {}\n".format(function.summary()) for function in self.functions]
            ),
        ]

        return "\n".join(l)

    def __str__(self):
        return self.string_rep()

    def __repr__(self):
        return self.string_rep()

    def printpretty(self):
        """Formatted print of model attributes"""
        print(self.string_rep())

    def make_dag(self, expand=set()):
        """Generate a DAG for the model"""
        G = nx.DiGraph()

        ## Inputs-to-Functions
        for f in self.functions:
            # Expand composed models
            if isinstance(f, FunctionModel) and (f.name in expand):
                G_ref = f.model.make_dag(expand=expand - {f})
                G_sub = nx.DiGraph()
                # Add nodes
                G_sub.add_node(f.name + ".var")
                G_sub.add_node(f.name + ".out")
                for g in f.model.functions:
                    G_sub.add_node(f.name + "." + g.name)
                # Add node metadata
                nx.set_node_attributes(G_sub, f.name, "parent")

                # Add edges
                for u, v, d in G_ref.edges(data=True):
                    # Add renamed edge
                    if u == "(var)":
                        G_sub.add_edge(f.name + ".var", f.name + "." + v, **d)
                    elif v == "(out)":
                        G_sub.add_edge(f.name + "." + u, f.name + ".out", **d)
                    else:
                        G_sub.add_edge(f.name + "." + u, f.name + "." + v, **d)

                # Compose the graphs
                G = nx.compose(G, G_sub)

            i_var = set(self.var).intersection(set(f.var))
            if len(i_var) > 0:
                s_var = "{}".format(i_var)
                if isinstance(f, FunctionModel) and (f.name in expand):
                    G.add_edge("(var)", f.name + ".var", label=s_var)
                else:
                    G.add_edge("(var)", f.name, label=s_var)

        ## Function-to-Function
        for i0 in range(len(self.functions)):
            for i1 in range(i0 + 1, len(self.functions)):
                f0 = self.functions[i0]
                f1 = self.functions[i1]
                i_var = set(f0.out).intersection(set(f1.var))

                ## If connected
                if len(i_var) > 0:
                    s_var = "{}".format(i_var)
                    ## Handle composed models
                    if isinstance(f0, FunctionModel) and (f0.name in expand):
                        name0 = f0.name + ".out"
                    else:
                        name0 = f0.name
                    if isinstance(f1, FunctionModel) and (f1.name in expand):
                        name1 = f1.name + ".out"
                    else:
                        name1 = f1.name

                    G.add_edge(name0, name1, label=s_var)

        ## Functions-to-Outputs
        for f in self.functions:
            i_out = set(self.out).intersection(set(f.out))

            if len(i_out) > 0:
                s_out = "{}".format(i_out)
                ## Target composed model's out
                if isinstance(f, FunctionModel) and (f.name in expand):
                    G.add_edge(f.name + ".out", "(out)", label=s_out)
                ## An ordinary function
                else:
                    G.add_edge(f.name, "(out)", label=s_out)

            # Add node metadata
            nx.set_node_attributes(G, {f.name: {"parent": self.name}})

        # Final metadata
        nx.set_node_attributes(G, {"(var)": {"parent": self.name}})
        nx.set_node_attributes(G, {"(out)": {"parent": self.name}})

        return G

    def show_dag(self, expand=set()):
        """Generate and show a DAG for the model"""
        from matplotlib.pyplot import show as pltshow

        G = self.make_dag(expand=expand)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## Plotting
            edge_labels = dict(
                [
                    (
                        (
                            u,
                            v,
                        ),
                        d["label"],
                    )
                    for u, v, d in G.edges(data=True)
                ]
            )
            n = G.size()

            ## Manual layout
            # if n == 2:
            if False:
                pos = {
                    "(var)": [-0.5, +0.5],
                    "(out)": [+0.5, -0.5],
                }
                pos[self.functions[0].name] = [+0.5, +0.5]
            ## Optimized layout
            else:
                try:
                    ## Planar, if possible
                    pos = nx.planar_layout(G)
                except nx.NetworkXException:
                    ## Scaled spring layout
                    pos = nx.spring_layout(
                        G,
                        k=0.6 * n,
                        pos={
                            "(Inputs)": [-0.5 * n, +0.5 * n],
                            "(Outputs)": [+0.5 * n, -0.5 * n],
                        },
                        fixed=["(var)", "(out)"],
                        threshold=1e-6,
                        iterations=100,
                    )

            # Generate colormap
            color_map = []
            for node in G:
                if G.nodes[node]["parent"] == self.name:
                    color_map.append("blue")
                else:
                    color_map.append("green")

            # Draw
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            nx.draw(G, pos, node_size=1000, with_labels=True, node_color=color_map)
            pltshow()
