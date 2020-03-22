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

    def test_ria(self):
        df_res = self.md >> gr.ev_form_ria(df_det="nom", limits=["g"])

        self.assertTrue(np.allclose(df_res["g"], [self.beta_true], atol=1e-3))

    def test_pma(self):
        df_res = self.md >> gr.ev_form_pma(df_det="nom", betas=dict(g=self.beta_true))

        self.assertTrue(np.allclose(df_res["g"], [0], atol=1e-3))
