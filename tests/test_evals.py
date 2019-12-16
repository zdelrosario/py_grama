import numpy as np
import pandas as pd
import unittest

from collections import OrderedDict as od
from context import core
from context import grama as gr

class TestModel(unittest.TestCase):
    """Test implementation of model_
    """

    def setUp(self):
        # 2D identity model with permuted df inputs
        domain_2d = gr.domain(
            bounds=od([("x", [-1., +1.]), ("y", [0., 1.])]),
        )

        self.model_2d = gr.model(
            functions=[
                gr.function(
                    lambda x: [x[0], x[1]],
                    ["x", "y"],
                    ["f", "g"],
                    "test"
                )
            ],
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
                        sign=-1,
                        d_name="uniform",
                        d_param={"loc": 0, "scale": 1}
                    )
                ]
            )
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

    def test_nominal_accurate(self):
        """Checks the nominal evaluation is accurate
        """
        df_res = gr.eval_nominal(self.model_2d)

        self.assertTrue(
            np.allclose(self.df_2d_nominal, df_res)
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
            np.allclose(df_grad, self.df_2d_grad)
        )

    def test_conservative_accurate(self):
        """Checks that conservative QE is accurate
        """
        df_res = gr.eval_conservative(
            self.model_2d,
            quantiles=[0.1, 0.1]
        )

        self.assertTrue(np.allclose(self.df_2d_qe, df_res))

## Run tests
if __name__ == "__main__":
    unittest.main()
