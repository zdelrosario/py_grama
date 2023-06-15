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
                fun=lambda x0, x1: self.beta_true * 2 - x0 - np.sqrt(3) * x1,
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

        ## Test other outputs
        limits = ["g_stress", "g_disp"]
        beta_names = ["beta_g_stress", "beta_g_disp"]
        # Return reliabilities
        df_rels = self.md_beam >> gr.ev_form_ria(
            df_det="nom", limits=limits, append=False, format="rels",
        )
        self.assertTrue(gr.df_equal(
            df_beam.apply(lambda col: norm.cdf(col) if col.name in beta_names else col)
                   .rename(columns={"beta_" + s: "rel_" + s for s in limits}),
            df_rels,
            close=True,
        ))
        # Return POFs
        df_pofs = self.md_beam >> gr.ev_form_ria(
            df_det="nom", limits=limits, append=False, format="pofs",
        )
        self.assertTrue(gr.df_equal(
            df_beam.apply(lambda col: 1-norm.cdf(col) if col.name in beta_names else col)
                   .rename(columns={"beta_" + s: "pof_" + s for s in limits}),
            df_pofs,
            close=True,
        ))

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

        ## Specify reliabilities
        df_rels = self.md_beam >> gr.ev_form_pma(
            df_det="nom", rels={"g_stress": norm.cdf(3), "g_disp": norm.cdf(3)}, append=False,
        )
        self.assertTrue(gr.df_equal(df_beam, df_rels, close=True))

        ## Specify POFs
        df_pofs = self.md_beam >> gr.ev_form_pma(
            df_det="nom", pofs={"g_stress": 1 - norm.cdf(3), "g_disp": 1 - norm.cdf(3)}, append=False,
        )
        self.assertTrue(gr.df_equal(df_beam, df_pofs, close=True))
