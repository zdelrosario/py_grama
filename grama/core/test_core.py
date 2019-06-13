import numpy as np
import pandas as pd
import unittest

from core import density_, domain_, model_, ev_df, pi

## Core function tests
##################################################
class TestPlumbing(unittest.TestCase):
    """test implementation of pipe and support functions
    """
    def setUp(self):
        self.model_default = model_()
        self.df_ok    = pd.DataFrame(data = {"x" : [0., 1.]})
        self.df_res   = pd.DataFrame(data = {"f" : [0., 1.]})

    ## Basic piping

    def test_model2eval(self):
        """Checks model evaluation via pipe
        """
        self.assertTrue(
            self.df_res.equals(
              self.model_default |pi| \
                ev_df(
                    df = self.df_ok,
                    append = False
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
        domain_2d = domain_(
            hypercube = True,
            inputs    = ["x", "y"],
            bounds    = {"x": [-1., +1.], "y": [0., 1.]},
            feasible  = lambda X: (-1 <= X[0]) * (x <= X[0]) * (0 <= X[1]) * (X[1] <= 1)
        )

        self.model_2d = model_(
            function = lambda x: [x[0], x[1]],
            outputs  = ["x", "y"],
            domain   = domain_2d,
            density  = None
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

class TestEvalDf(unittest.TestCase):
    """Test implementation of ev_df()
    """

    def test_catch_no_df(self):
        """Checks that ev_df() raises when no input df is given.
        """
        self.assertRaises(
            ValueError,
            ev_df,
            model_()
        )

## Run tests
if __name__ == "__main__":
    unittest.main()
