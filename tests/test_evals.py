import numpy as np
import pandas as pd
import unittest

from context import core
from context import grama as gr
from context import pi

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
            bounds    = {"x": [0., +1.], "y": [0., +1.]}
        )

        self.model_2d = core.model_(
            function = lambda x: x[0] + x[1],
            outputs  = ["f"],
            domain   = domain_2d,
            density  = core.density_(
                pdf = lambda x: 1,
                pdf_factors = ["uniform", "uniform"],
                pdf_param = [
                    {"loc": 0, "scale": 1},
                    {"loc": 0, "scale": 1}
                ],
                pdf_qt_sign = [ 0, -1]
            )
        )
        self.df_2d = pd.DataFrame(data = {"y": [0.], "x": [+1.]})
        self.res_2d = self.model_2d.evaluate(self.df_2d)
        self.df_2d_nominal = pd.DataFrame(data = {"x": [0.5], "y": [0.5], "f": [1.0]})
        self.df_2d_qe = pd.DataFrame(data = {"x": [0.5], "y": [0.1], "f": [0.6]})

    ## Test default evaluations

    def test_nominal_accurate(self):
        """Checks the nominal evaluation is accurate
        """
        self.assertTrue(
            self.df_2d_nominal.equals(
                self.model_2d |pi| gr.ev_nominal()
            )
        )

    def test_grad_fd_accurate(self):
        """Checks the FD is accurate
        """
        df_grad = self.model_2d |pi| \
                   gr.ev_grad_fd(df_base = self.df_2d_nominal, append = False)
        grad_hat = df_grad.values

        self.assertTrue(
            # np.linalg.norm(grad_hat - np.array([1, 1])) / np.sqrt(2) <= eps_fd
            np.allclose(grad_hat, np.array([1, 1]))
        )

    def test_conservative_accurate(self):
        """Checks that conservative QE is accurate
        """
        df_res = self.model_2d |pi| gr.ev_conservative(quantiles = [0.1, 0.1])

        self.assertTrue(
            np.allclose(self.df_2d_qe.values, df_res.values)
        )

## Run tests
if __name__ == "__main__":
    unittest.main()
