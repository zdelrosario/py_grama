import numpy as np
import pandas as pd
import unittest

from context import grama as gr
from context import models


class TestPlumbing(unittest.TestCase):
    """Test implementation of pipe-shortcut functions

    Note that these do not check correctness of the original function,
    just that the piped version matches the non-piped version.
    """

    def setUp(self):
        self.model_default = models.make_test()
        self.df_test = models.df_test_input

    ## Evaluations
    # --------------------------------------------------

    def test_ev_df(self):
        """Check ev_df()
        """
        df_res = gr.eval_df(self.model_default, df=self.df_test)

        self.assertTrue(
            gr.df_equal(df_res, self.model_default >> gr.ev_df(df=self.df_test))
        )

    def test_ev_nominal(self):
        """Check ev_nominal()
        """
        df_res = gr.eval_nominal(self.model_default, df_det="nom")

        self.assertTrue(
            gr.df_equal(df_res, self.model_default >> gr.ev_nominal(df_det="nom"))
        )

    def test_ev_grad_fd(self):
        """Check ev_grad_fd()
        """
        df_res = gr.eval_grad_fd(self.model_default, df_base=self.df_test)

        self.assertTrue(
            gr.df_equal(
                df_res, self.model_default >> gr.ev_grad_fd(df_base=self.df_test)
            )
        )

    def test_ev_conservative(self):
        """Check ev_conservative()
        """
        df_res = gr.eval_conservative(self.model_default, df_det="nom")

        self.assertTrue(
            gr.df_equal(df_res, self.model_default >> gr.ev_conservative(df_det="nom"))
        )

    def test_ev_sample(self):
        """Check ev_sample()
        """
        df_res = gr.eval_sample(self.model_default, n=1, seed=101, df_det="nom")

        self.assertTrue(
            gr.df_equal(
                df_res,
                self.model_default >> gr.ev_sample(seed=101, n=1, df_det="nom")
            )
        )

    def test_ev_sinews(self):
        """Check ev_sinews()
        """
        df_res = gr.eval_sinews(self.model_default, seed=101, df_det="nom")

        self.assertTrue(
            gr.df_equal(
                df_res, self.model_default >> gr.ev_sinews(seed=101, df_det="nom")
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
