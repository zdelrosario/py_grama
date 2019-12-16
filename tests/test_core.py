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
        domain_2d = gr.Domain(
            bounds=dict([("x", [-1., +1.]), ("y", [0., 1.])]),
        )

        self.model_2d = gr.Model(
            functions=[
                gr.Function(
                    lambda x: [x[0], x[1]],
                    ["x", "y"],
                    ["x", "y"],
                    "test"
                )
            ],
            domain=domain_2d,
            density=gr.Density(
                marginals=[
                    gr.MarginalNamed(
                        "x",
                        d_name="uniform",
                        d_param={"loc":-1, "scale": 2}
                    ),
                    gr.MarginalNamed(
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
