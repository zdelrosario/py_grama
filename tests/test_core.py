import numpy as np
import pandas as pd
import unittest

from context import core

## Core function tests
##################################################
class TestModel(unittest.TestCase):
    """Test implementation of model_
    """

    def setUp(self):
        # Default model
        self.model_default = core.model_()
        self.df_wrong = pd.DataFrame(data = {"y" : [0., 1.]})
        self.df_ok    = pd.DataFrame(data = {"x" : [0., 1.]})

        # 2D identity model with permuted df inputs
        domain_2d = core.domain_(
            hypercube = True,
            inputs    = ["x", "y"],
            bounds    = {"x": [-1., +1.], "y": [0., 1.]},
            feasible  = lambda X: (-1 <= X[0]) * (x <= X[0]) * (0 <= X[1]) * (X[1] <= 1)
        )

        self.model_2d = core.model_(
            function = lambda x: [x[0], x[1]],
            outputs  = ["x", "y"],
            domain   = domain_2d,
            density  = core.density_(
                pdf = lambda x: 1,
                pdf_factors = ["uniform", "uniform"],
                pdf_param = [
                    {"loc":-1, "scale": 2},
                    {"loc": 0, "scale": 1}
                ]
            )
        )
        self.model_2d_corr = core.model_(
            function = lambda x: [x[0], x[1]],
            outputs  = ["x", "y"],
            domain   = domain_2d,
            density  = core.density_(
                pdf = lambda x: 1,
                pdf_factors = ["uniform", "uniform"],
                pdf_param = [
                    {"loc":-1, "scale": 2},
                    {"loc": 0, "scale": 1}
                ],
                pdf_corr = [0.5]
            )
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

    ## Test quantile evaluation

    def test_quantile(self):
        """Checks that model_.sample_quantile() evaluates correctly.
        """
        self.assertTrue(
            np.all(
                self.model_2d.sample_quantile(np.array([[0.5, 0.5]])) == \
                np.array([0.0, 0.5])
            )
        )

    def test_quantile_corr(self):
        """Checks that model_.sample_quantile() evaluates correctly with copula model.
        """
        self.assertTrue(
            np.all(
                self.model_2d_corr.sample_quantile(np.array([[0.5, 0.5]])) == \
                np.array([0.0, 0.5])
            )
        )

class TestVectorizedModel(unittest.TestCase):
    """Test the implementation of model_vectorized_
    """
    def setUp(self):
        func = lambda df: pd.DataFrame(data = {"f": df['x']})

        self.df_input = pd.DataFrame(data = {"x": [0, 1, 2]})
        self.df_output = pd.DataFrame(data = {"f": [0, 1, 2]})
        self.model_vectorized = core.model_vectorized_(
            function=func,
            outputs=["f"]
        )

    def test_copy(self):
        """Invoke a copy through pipe evaluation
        """
        df_res = self.model_vectorized >> \
            core.ev_df(df=self.df_input, append=False)

        self.assertTrue(self.df_output.equals(df_res))

class TestEvalDf(unittest.TestCase):
    """Test implementation of eval_df()
    """

    def test_catch_no_df(self):
        """Checks that eval_df() raises when no input df is given.
        """
        self.assertRaises(
            ValueError,
            core.eval_df,
            core.model_()
        )

## Run tests
if __name__ == "__main__":
    unittest.main()
