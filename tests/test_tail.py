import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest
import networkx as nx

from context import grama as gr
from context import models


class TestFORM(unittest.TestCase):
    """Test implementations of FORM
    """

    def setUp(self):
        ## Linear limit state w/ MPP off initial guess
        self.beta_true = 3
        self.md = (
            gr.Model()
            >> gr.cp_function(
                fun=lambda x: self.beta_true * 2 - x[0] - np.sqrt(3) * x[1],
                var=2,
                out=["g"],
            )
            >> gr.cp_marginals(
                x0=dict(dist="norm", loc=0, scale=1, sign=1),
                x1=dict(dist="norm", loc=0, scale=1, sign=1),
            )
            >> gr.cp_copula_independence()
        )

        ## Linear limit state w/ lognormal marginals
        self.md_log = (
            gr.Model()
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(
                    g=gr.exp(gr.sqrt(2) * 1) - df.x * df.y
                ),
                var=["x", "y"],
                out=["g"]
            )
            >> gr.cp_marginals(
                x=dict(dist="lognorm", loc=0, scale=1, s=1),
                y=dict(dist="lognorm", loc=0, scale=1, s=1),
            )
            >> gr.cp_copula_independence()
        )
        self.df_mpp = gr.df_make(
            x=gr.exp(gr.sqrt(2)/2),
            y=gr.exp(gr.sqrt(2)/2),
            beta_g=1.0,
            g=0.0,
        )

        ## Cantilever beam for flatten test
        self.md_beam = models.make_cantilever_beam()

    def test_ria(self):
        ## Test accuracy
        df_res = self.md >> gr.ev_form_ria(df_det="nom", limits=["g"])
        self.assertTrue(np.allclose(df_res["beta_g"], [self.beta_true], atol=1e-3))

        ## Test MPP mapped correctly
        df_mpp = self.md_log >> gr.ev_form_ria(df_det="nom", limits=["g"])
        self.assertTrue(
            gr.df_equal(df_mpp, self.df_mpp[["x", "y", "beta_g"]], close=True)
        )

        ## Test flatten
        df_beam = self.md_beam >> gr.ev_form_ria(
            df_det="nom", limits=["g_stress", "g_disp"], append=False
        )
        self.assertTrue(df_beam.shape[0] == 1)

    def test_pma(self):
        ## Test accuracy
        df_res = self.md >> gr.ev_form_pma(df_det="nom", betas=dict(g=self.beta_true))
        self.assertTrue(np.allclose(df_res["g"], [0], atol=1e-3))

        ## Test MPP mapped correctly
        df_mpp = self.md_log >> gr.ev_form_pma(df_det="nom", betas=dict(g=1.0))
        self.assertTrue(
            gr.df_equal(df_mpp, self.df_mpp[["x", "y", "g"]], close=True)
        )

        ## Test flatten
        df_beam = self.md_beam >> gr.ev_form_pma(
            df_det="nom", betas={"g_stress": 3, "g_disp": 3}, append=False,
        )
        self.assertTrue(df_beam.shape[0] == 1)
