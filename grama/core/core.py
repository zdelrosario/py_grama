## grama core functions
# Zachary del Rosario, March 2019

import numpy as np
import pandas as pd

from functools import partial
from toolz import curry

## Helper functions
##################################################
# Infix to help define pipe
class Infix(object):
    def __init__(self, func):
        self.func = func
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return Infix(partial(self.func, other))
    def __call__(self, v1, v2):
        return self.func(v1, v2)

# Pipe function
@Infix
def pi(x, f):
    """Infix pipe operator.
    """
    return f(x)

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
        self.hypercube = hypercube
        self.inputs    = inputs
        self.bounds    = bounds
        self.feasible  = feasible

# Density parent class
class density_:
    """Parent class for join densities
    """
    def __init__(
            self,
            pdf = lambda x: 0.5,
            pdf_factors = ["unif"],
            pdf_param = [{"lower": -1., "upper": +1.}]
    ):
        """Initialize

        @param pdf density function \rho(x) : R^n_in -> R
        @param pdf_factors if joint density can be factored, list of names
               of marginal distributions
        @param pdf_param if joint density can be factored, list of dict
               of margin density parameters

        @pre (len(pdf_factors) == n_in) || (pdf_factors is None)
        @pre (len(pdf_param) == n_in) || (pdf_param is None)
        """
        self.pdf         = pdf
        self.pdf_factors = pdf_factors
        self.pdf_param   = pdf_param

# Model parent class
class model_:
    """Parent class for grama models.
    """

    def __init__(
            self,
            function = lambda x: x,
            outputs  = ["f"],
            domain   = domain_(),
            density  = density_(),
    ):
        """Constructor

        @param function defining the model mapping f(x) : R^n_in -> R^n_out
        @param inputs to function; ordering of abstract inputs x given by inputs
        @param ouputs of function outputs
        @param domain object of class domain_
        @param density object of class density_ or None

        @pre len(domain.inputs) == n_in
        @pre len(outputs) == n_out
        @pre isinstance(domain, domain_)
        @pre isinstance(density, density_) || (density is None)

        Default model is 1D identity over the interval [-1, +1] with a uniform density.
        """
        self.function = function
        self.outputs  = outputs
        self.domain   = domain
        self.density  = density

        ## Convenience constants
        self.n_in  = len(self.domain.inputs)
        self.n_out = len(outputs)

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
        return pd.DataFrame(data = results, columns = self.outputs)

    def printpretty(self):
        """Formatted print of model attributes
        """

## Default pipeline evaluation function
@curry
def eval_df(model, df = None, append = True):
    """Evaluates a given model at a given dataframe

    @param df input dataframe to evaluate (Pandas.DataFrame)
    @param append bool flag; append results to original dataframe?
    """

    if df is None:
        raise ValueError("No input df given!")

    df_res = model.evaluate(df)

    if append:
        df_res = pd.concat([df.reset_index(drop = True), df_res], axis=1)

    return df_res
