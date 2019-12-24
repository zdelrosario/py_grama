import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

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
        domain_2d = gr.Domain(bounds={"x": [-1., +1.], "y": [0., 1.]})
        marginals = {}
        marginals["x"] = gr.MarginalNamed(
            d_name="uniform",
            d_param={"loc":-1, "scale": 2}
        )
        marginals["y"] = gr.MarginalNamed(
            sign=-1,
            d_name="uniform",
            d_param={"loc": 0, "scale": 1}
        )

        self.model_2d = gr.Model(
            functions=[
                gr.Function(
                    lambda x: [x[0], x[1]],
                    ["x", "y"],
                    ["x", "y"],
                    "test",
                    0
                )
            ],
            domain=domain_2d,
            density=gr.Density(marginals=marginals)
        )
        self.df_2d = pd.DataFrame(data = {"y": [0.], "x": [+1.]})
        self.res_2d = self.model_2d.evaluate_df(self.df_2d)

        self.df_median_in = pd.DataFrame({"x": [0.5], "y": [0.5]})
        self.df_median_out = pd.DataFrame({"x": [0.0], "y": [0.5]})

        self.model_3d = gr.Model(
            functions=[
                gr.Function(
                    lambda x: x[0] + x[1] + x[2],
                    ["x", "y", "z"],
                    ["f"],
                    "test",
                    0
                )
            ],
            density=gr.Density(marginals=marginals)
        )

        ## Timing check
        self.model_slow = gr.Model(
            functions=[
                gr.Function(
                    lambda x: x,
                    ["x"],
                    ["y"],
                    "f0",
                    1
                ),
                gr.Function(
                    lambda x: x,
                    ["x"],
                    ["y"],
                    "f1",
                    1
                )
            ]
        )

    def test_prints(self):
        ## Invoke printpretty
        self.model_3d.printpretty()

    def test_timings(self):
        ## Default is zero
        self.assertTrue(self.model_2d.runtime(1) == 0)

        ## Estimation accounts for both functions
        self.assertTrue(np.allclose(self.model_slow.runtime(1), 2))

        ## Fast function has empty message
        self.assertTrue(self.model_2d.runtime_message(self.df_2d) is None)

        ## Slow function returns string message
        msg = self.model_slow.runtime_message(pd.DataFrame({"x": [0]}))
        self.assertTrue(
            isinstance(msg, str)
        )

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

    def test_var_outer(self):
        with self.assertRaises(ValueError):
            self.model_3d.var_outer(self.df_2d)

        with self.assertRaises(ValueError):
            self.model_3d.var_outer(self.df_2d, df_det="foo")

        with self.assertRaises(ValueError):
            self.model_3d.var_outer(self.df_2d, df_det=self.df_2d)

    def test_var_rand_quantile(self):
        ## TODO: Copula tests

        with self.assertRaises(ValueError):
            self.model_2d.var_rand_quantile(self.df_wrong)

    ## Test re-ordering issues

    def test_2d_output_names(self):
        """Checks that proper output names are assigned to resulting DataFrame
        """
        self.assertEqual(
            set(self.model_2d.evaluate_df(self.df_2d).columns),
            set(self.model_2d.out)
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

class TestMarginal(unittest.TestCase):

    def setUp(self):
        self.marginal_named = gr.MarginalNamed(
            d_name="norm",
            d_param={"loc": 0, "scale": 1}
        )

    def test_fcn(self):

        ## Invoke summary
        self.marginal_named.summary()

        self.assertTrue(
            self.marginal_named.l(0.5) == norm.pdf(0.5)
        )

        self.assertTrue(
            self.marginal_named.p(0.5) == norm.cdf(0.5)
        )

        self.assertTrue(
            self.marginal_named.q(0.5) == norm.ppf(0.5)
        )


class TestDomain(unittest.TestCase):

    def setUp(self):
        self.domain = gr.Domain(bounds={"x": (0, 1)})

    def test_blank(self):
        ## Test blank domain valid
        gr.Domain()

        ## Invoke summary
        self.domain.bound_summary("x")

        ## Invoke summary;
        self.assertTrue(
            self.domain.bound_summary("y").find("unbounded") > -1
        )

class TestFunction(unittest.TestCase):

    def setUp(self):
        self.fcn = gr.Function(
            lambda x: x,
            ["x"],
            ["x"],
            "test",
            0
        )

        self.fcn_vec = gr.FunctionVectorized(
            lambda df: df,
            ["x"],
            ["x"],
            "test",
            0
        )

        self.df = pd.DataFrame({"x": [0]})

        self.df_wrong = pd.DataFrame({"z": [0]})

    def test_function(self):
        fcn_copy = self.fcn.copy()

        self.assertTrue(self.fcn.var == fcn_copy.var)
        self.assertTrue(self.fcn.out == fcn_copy.out)
        self.assertTrue(self.fcn.name == fcn_copy.name)

        pd.testing.assert_frame_equal(
            self.df,
            self.fcn.eval(self.df),
            check_dtype=False
        )

        with self.assertRaises(ValueError):
            self.fcn.eval(self.df_wrong)

        ## Invoke summary
        self.fcn.summary()

    def test_function_vectorized(self):
        fcn_copy = self.fcn_vec.copy()

        self.assertTrue(self.fcn_vec.var == fcn_copy.var)
        self.assertTrue(self.fcn_vec.out == fcn_copy.out)
        self.assertTrue(self.fcn_vec.name == fcn_copy.name)

        pd.testing.assert_frame_equal(
            self.df,
            self.fcn_vec.eval(self.df),
            check_dtype=False
        )


## Run tests
if __name__ == "__main__":
    unittest.main()
