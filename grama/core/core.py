## grama core functions
# Zachary del Rosario, March 2019

import numpy as np
import pandas as pd
import unittest

from toolz import curry

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

## Core function tests
##################################################
class TestPlumbing(unittest.TestCase):
    """test implementation of pipe and support functions
    """
    def setUp(self):
        self.model_default = model_()
        self.df_ok    = pd.DataFrame(data = {"x" : [0., 1.]})

    ## Basic piping

    def test_model2eval(self):
        """Checks model evaluation via pipe
        """
        self.assertTrue(
            self.df_ok.equals(
              self.model_default |pi| \
                eval_df(
                    df = self.df_ok
                )
            )
        )

class TestModel(unittest.TestCase):
    """Test implementation of model_
    """

    def setUp(self):
        # Default model
        self.model_default = model_()
        self.df_wrong = pd.DataFrame(data = {"y" : [0., 1.]})
        self.df_ok    = pd.DataFrame(data = {"x" : [0., 1.]})

        # 2D identity model with permuted df inputs
        self.model_2d = model_(
            function = lambda x: [x[0], x[1]],
            inputs   = ["x", "y"],
            outputs  = ["x", "y"],
            domain   = {"x": [-1., +1.], "y": [0., 1.]},
            density  = lambda x: [0.5, 0.5]
        )
        self.df_2d = pd.DataFrame(data = {"y": [0.], "x": [+1.]})
        self.res_2d = self.model_2d.evaluate(self.df_2d)

    ## Basic functionality with default arguments

    def test_catch_input_mismatch(self):
        """Checks that proper exception is thrown if evaluate(df) passed a DataFrame
        without the proper columns.
        """
        self.assertRaises(
            ValueError,
            self.model_default.evaluate,
            self.df_wrong
        )

    ## Test re-ordering issues

    def test_2d_output_names(self):
        """Checks that proper output names are assigned to resulting DataFrame
        """
        self.assertEqual(
            set(self.model_2d.evaluate(self.df_2d).columns),
            set(self.model_2d.outputs)
        )

    def test_2d_identity(self):
        """Checks that re-ordering of inputs handled properly
        """
        self.assertTrue(
            self.df_2d.equals(
                self.res_2d.loc[:, self.df_2d.columns]
            )
        )

## Run tests
if __name__ == "__main__":
    unittest.main()
