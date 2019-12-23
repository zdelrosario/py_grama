import numpy as np
import pandas as pd
import unittest

from collections import OrderedDict as od
from context import core
from context import grama as gr
from pyDOE import lhs

##################################################
class TestDefaults(unittest.TestCase):

    def setUp(self):
        # 2D identity model with permuted df inputs
        domain_2d = gr.Domain(bounds={"x": [-1., +1], "y": [0., 1.]})
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
                    ["f", "g"],
                    "test"
                )
            ],
            domain=domain_2d,
            density=gr.Density(marginals=marginals)
        )

        ## Correct results
        self.df_2d_nominal = pd.DataFrame(
            data={"x": [0.0], "y": [0.5], "f": [0.0], "g": [0.5]}
        )
        self.df_2d_grad = pd.DataFrame(
            data={"Df_Dx": [1.0], "Dg_Dx": [0.0], "Df_Dy": [0.0], "Dg_Dy": [1.0]}
        )
        self.df_2d_qe = pd.DataFrame(
            data={"x": [0.0], "y": [0.1], "f": [0.0], "g": [0.1]}
        )

    ## Test default evaluations

    def test_nominal(self):
        """Checks the nominal evaluation is accurate
        """
        df_res = gr.eval_nominal(self.model_2d)

        ## Accurate
        self.assertTrue(
            np.allclose(self.df_2d_nominal, df_res)
        )

        ## Pass-through
        self.assertTrue(
            np.allclose(
                self.df_2d_nominal.drop(["f", "g"], axis=1),
                gr.eval_nominal(self.model_2d, skip=True)
            )
        )

    def test_grad_fd_accurate(self):
        """Checks the FD is accurate
        """
        df_grad = gr.eval_grad_fd(
            self.model_2d,
            df_base=self.df_2d_nominal,
            append=False
        )

        self.assertTrue(
            np.allclose(df_grad[self.df_2d_grad.columns], self.df_2d_grad)
        )

    def test_conservative(self):
        ## Accuracy
        df_res = gr.eval_conservative(
            self.model_2d,
            quantiles=[0.1, 0.1]
        )

        self.assertTrue(np.allclose(self.df_2d_qe, df_res))

        ## Repeat scalar value
        self.assertTrue(np.allclose(
            self.df_2d_qe,
            gr.eval_conservative(self.model_2d, quantiles=0.1)
        ))

        ## Pass-through
        self.assertTrue(np.allclose(
            self.df_2d_qe.drop(["f", "g"], axis=1),
            gr.eval_conservative(self.model_2d, quantiles=0.1, skip=True)
        ))

##################################################
class TestRandomSampling(unittest.TestCase):
    def setUp(self):
        self.md = gr.Model() >> \
                  gr.cp_function(fun=lambda x: x, var=1, out=1) >> \
                  gr.cp_marginals(x0={"dist": "uniform", "loc": 0, "scale": 1})

        self.md_2d = gr.Model() >> \
                  gr.cp_function(fun=lambda x: x[0], var=2, out=1) >> \
                  gr.cp_marginals(
                      x0={"dist": "uniform", "loc": 0, "scale": 1},
                      x1={"dist": "uniform", "loc": 0, "scale": 1}
                  )

    def test_lhs(self):
        ## Accurate
        n=2
        df_res = gr.eval_lhs(self.md_2d, n=n, df_det="nom", seed=101)

        np.random.seed(101)
        df_truth = pd.DataFrame(data=lhs(2, samples=n), columns=["x0", "x1"])
        df_truth["y0"] = df_truth["x0"]

        pd.testing.assert_frame_equal(
            df_res,
            df_truth,
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )

        ## Rounding
        df_round = gr.eval_lhs(self.md_2d, n=n+0.1, df_det="nom", seed=101)

        pd.testing.assert_frame_equal(
            df_round,
            df_truth,
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )

        ## Pass-through
        df_pass = gr.eval_lhs(self.md_2d, n=n, skip=True, df_det="nom", seed=101)

        pd.testing.assert_frame_equal(
            df_pass,
            df_truth[["x0", "x1"]],
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )

    def test_monte_carlo(self):
        ## Accurate
        n=2
        df_res = gr.eval_monte_carlo(self.md, n=n, df_det="nom", seed=101)

        np.random.seed(101)
        df_truth = pd.DataFrame({"x0": np.random.random(n)})
        df_truth["y0"] = df_truth["x0"]

        pd.testing.assert_frame_equal(
            df_res,
            df_truth,
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )

        ## Rounding
        df_round = gr.eval_monte_carlo(self.md, n=n+0.1, df_det="nom", seed=101)

        pd.testing.assert_frame_equal(
            df_round,
            df_truth,
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )

        ## Pass-through
        df_pass = gr.eval_monte_carlo(
            self.md,
            n=n,
            skip=True,
            df_det="nom",
            seed=101
        )

        pd.testing.assert_frame_equal(
            df_pass[["x0"]],
            df_truth[["x0"]],
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )


## Run tests
if __name__ == "__main__":
    unittest.main()
