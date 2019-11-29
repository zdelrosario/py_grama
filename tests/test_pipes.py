import numpy as np
import pandas as pd
import unittest

from context import grama as gr

class TestPlumbing(unittest.TestCase):
    """Test implementation of pipe-shortcut functions

    Note that these do not check correctness of the original function,
    just that the piped version matches the non-piped version.
    """
    def setUp(self):
        self.model_default = gr.model_()
        self.df_ok         = pd.DataFrame(data = {"x" : [0., 1.]})

    ## Evaluations
    # --------------------------------------------------

    def test_ev_df(self):
        """Check ev_df()
        """
        df_res = gr.eval_df(self.model_default, df=self.df_ok)

        self.assertTrue(
            df_res.equals(
              self.model_default >> gr.ev_df(df=self.df_ok)
            )
        )

    def test_ev_nominal(self):
        """Check ev_nominal()
        """
        df_res = gr.eval_nominal(self.model_default)

        self.assertTrue(
            df_res.equals(
              self.model_default >> gr.ev_nominal()
            )
        )

    def test_ev_grad_fd(self):
        """Check ev_grad_fd()
        """
        df_res = gr.eval_grad_fd(self.model_default, df_base=self.df_ok)

        self.assertTrue(
            df_res.equals(
              self.model_default >> gr.ev_grad_fd(df_base=self.df_ok)
            )
        )

    def test_ev_conservative(self):
        """Check ev_conservative()
        """
        df_res = gr.eval_conservative(self.model_default)

        self.assertTrue(
            df_res.equals(
              self.model_default >> gr.ev_conservative()
            )
        )

    def test_ev_monte_carlo(self):
        """Check ev_monte_carlo()
        """
        df_res = gr.eval_monte_carlo(self.model_default, seed=101)

        self.assertTrue(
            df_res.equals(
              self.model_default >> gr.ev_monte_carlo(seed=101)
            )
        )

    def test_ev_lhs(self):
        """Check ev_lhs()
        """
        df_res = gr.eval_lhs(self.model_default, seed=101)

        self.assertTrue(
            df_res.equals(
              self.model_default >> gr.ev_lhs(seed=101)
            )
        )

    def test_ev_sinews(self):
        """Check ev_sinews()
        """
        df_res = gr.eval_sinews(self.model_default, seed=101)

        self.assertTrue(
            df_res.equals(
              self.model_default >> gr.ev_sinews(seed=101)
            )
        )

    ## Fittings
    # --------------------------------------------------

    ## TODO

    ## Compositions
    # --------------------------------------------------

    ## TODO

    ## Transforms
    # --------------------------------------------------

    ## TODO

## Run tests
if __name__ == "__main__":
    unittest.main()
