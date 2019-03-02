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
# Model parent class
class model_:
    """Parent class for grama models.
    """

    def __init__(
            self,
            function = lambda x: x,
            inputs   = ["x"],
            outputs  = ["x"],
            domain   = {"x": [-1, +1]},
            density  = lambda x: 0.5,
    ):
        """Constructor

        All arguments must satisfy particular invariants:
        @param function must be a function f(x) : R^n_in -> R^n_out
        @param inputs must satisfy len(inputs) == n_in; ordering of abstract inputs x must
               match names given by inputs
        @param ouputs must satisfy len(inputs) == n_out; ordering of abstract outputs f(x) must
               match names given by outputs
        @param domain must be a dictionary with set(domain.keys()) == set(inputs)
        @param density must be a function rho(x) : R^d -> R^n_out

        Default model is 1D identity over the interval [-1, +1] with a uniform density.
        """
        self.function = function
        self.inputs   = inputs
        self.outputs  = outputs
        self.domain   = domain
        self.density  = density

        ## Convenience constants
        self.n_in  = len(inputs)
        self.n_out = len(outputs)

    def evaluate(self, df):
        """Evaluate function using an input dataframe

        Does not assume a vectorized function.
        """

        ## Check invariant; model inputs must be subset of df columns
        if not set(self.inputs).issubset(set(df.columns)):
            raise ValueError("Model inputs not a subset of given columns")

        ## Set up output
        n_rows  = df.shape[0]
        results = np.zeros((n_rows, self.n_out))
        for ind in range(n_rows):
            results[ind] = self.function(df.loc[ind, self.inputs])

        ## Package output as DataFrame
        return pd.DataFrame(data = results, columns = self.outputs)

## Default pipeline evaluation function
@curry
def eval_df(model, df = None):
    """Evaluates a given model at a given dataframe
    """
    return model.evaluate(df)
