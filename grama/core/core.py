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
            pdf_corr    = None
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

        @pre (len(pdf_factors) == n_in) || (pdf_factors is None)
        @pre (len(pdf_param) == n_in) || (pdf_param is None)
        @pre (len(pdf_corr == len(np.triu_indices(n_in, 1)[0]))) || (pdf_param is None)
        """
        self.pdf         = pdf if (pdf is not None) else lambda x: 0.5
        self.pdf_factors = pdf_factors if (pdf_factors is not None) else ["unif"]
        self.pdf_param   = pdf_param if (pdf_param is not None) else [
            {"lower": -1., "upper": +1.}
        ]
        self.pdf_corr    = pdf_corr if (pdf_corr is not None) else None

# Model parent class
class model_:
    """Parent class for grama models.
    """

    def __init__(
            self,
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
        return pd.DataFrame(data = results, columns = self.outputs)

    def printpretty(self):
        """Formatted print of model attributes
        """

class model_df_(model_):
    """Derived class for grama models.

    Given function must be vectorized over dataframes
    """

    def evaluate(self, df):
        """Evaluate function using an input dataframe

        Assumes function is vectorized over dataframes.
        """
        return self.function(df)

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
