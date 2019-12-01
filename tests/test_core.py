import numpy as np
import pandas as pd
import unittest

from collections import OrderedDict as od
from context import grama as gr
from context import models

## Core function tests
##################################################
class TestModel(unittest.TestCase):
    """Test implementation of model
    """

    def setUp(self):
        # Default model
        self.df_wrong = pd.DataFrame(data={"z" : [0., 1.]})

        # 2D identity model with permuted df inputs
        domain_2d = gr.domain(
            bounds=od([("x", [-1., +1.]), ("y", [0., 1.])]),
        )

        self.model_2d = gr.model(
            function=lambda x: [x[0], x[1]],
            outputs=["x", "y"],
            domain=domain_2d,
            density=gr.density(
                marginals=[
                    gr.marginal_named(
                        "x",
                        d_name="uniform",
                        d_param={"loc":-1, "scale": 2}
                    ),
                    gr.marginal_named(
                        "y",
                        d_name="uniform",
                        d_param={"loc": 0, "scale": 1}
                    )
                ]
            )
        )
        self.df_2d = pd.DataFrame(data = {"y": [0.], "x": [+1.]})
        self.res_2d = self.model_2d.evaluate_df(self.df_2d)

        self.df_median_in = pd.DataFrame({"x": [0.5], "y": [0.5]})
        self.df_median_out = pd.DataFrame({"x": [0.0], "y": [0.5]})

    ## Basic functionality with default arguments

    def test_catch_input_mismatch(self):
        """Checks that proper exception is thrown if evaluate(df) passed a
        DataFrame without the proper columns.
        """
        self.assertRaises(
            ValueError,
            self.model_2d.evaluate_df,
            self.df_wrong
        )

    ## Test re-ordering issues

    def test_2d_output_names(self):
        """Checks that proper output names are assigned to resulting DataFrame
        """
        self.assertEqual(
            set(self.model_2d.evaluate_df(self.df_2d).columns),
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
        """Checks that model.sample_quantile() evaluates correctly.
        """
        self.assertTrue(
            self.model_2d.var_rand_quantile(self.df_median_in).equals(
                self.df_median_out
            )
        )

    ## TODO: Once copula model implemented
    # def test_quantile_corr(self):
    #     """Checks that model.sample_quantile() evaluates correctly with copula model.
    #     """
    #     self.assertTrue(
    #         np.all(
    #             self.model_2d_corr.sample_quantile(np.array([[0.5, 0.5]])) == \
    #             np.array([0.0, 0.5])
    #         )
    #     )

class TestVectorizedModel(unittest.TestCase):
    """Test the implementation of model_vectorized
    """
    def setUp(self):
        func = lambda df: pd.DataFrame(data = {"f": df['x']})

        self.df_input = pd.DataFrame(data = {"x": [0, 1, 2]})
        self.df_output = pd.DataFrame(data = {"f": [0, 1, 2]})
        self.model_vectorized = gr.model_vectorized(
            function=func,
            outputs=["f"],
            domain=gr.domain(bounds=od([("x", [0, 2])])),
            density=gr.density(marginals=[gr.marginal_named("x")])
        )

    def test_copy(self):
        """Invoke a copy through pipe evaluation
        """
        df_res = self.model_vectorized >> \
            gr.ev_df(df=self.df_input, append=False)

        self.assertTrue(self.df_output.equals(df_res))

class TestEvalDf(unittest.TestCase):
    """Test implementation of eval_df()
    """
    def setUp(self):
        self.model = models.make_test()

    def test_catch_no_df(self):
        """Checks that eval_df() raises when no input df is given.
        """
        self.assertRaises(
            ValueError,
            gr.eval_df,
            self.model
        )

## Run tests
if __name__ == "__main__":
    unittest.main()
