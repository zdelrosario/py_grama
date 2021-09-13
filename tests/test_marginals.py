import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import models, data

## Marginals function tests
##################################################
class TestMarginalTools(unittest.TestCase):
    def setUp(self):
        self.mg_gkde = gr.marg_gkde(data.df_stang.E)
        self.mg_norm = gr.marg_named(data.df_stang.E, "norm")

    def test_marginals(self):
        median = np.median(data.df_stang.E)

        l_gkde = self.mg_gkde.l(np.array([1, 10000, 10400, 10800, 1e6]))
        p_gkde = self.mg_gkde.p(np.array([1, 10000, 10400, 10800, 1e6]))
        q_gkde = self.mg_gkde.q(np.array([0.0, 0.25, 0.50, 0.75, 1.0]))
        self.mg_gkde.summary()

        self.assertTrue(np.isclose(q_gkde[2], median, atol=0, rtol=0.05))

        l_norm = self.mg_norm.l(np.array([10000, 10400, 10800]))
        p_norm = self.mg_norm.p(np.array([10000, 10400, 10800]))
        q_norm = self.mg_norm.q(np.array([0.25, 0.50, 0.75]))
        self.mg_norm.summary()

        self.assertTrue(np.isclose(q_norm[1], median, atol=0, rtol=0.05))

        ## Raises error when dataframe passed
        with self.assertRaises(ValueError):
            gr.marg_named(data.df_stang, "norm")
        with self.assertRaises(ValueError):
            gr.marg_gkde(data.df_stang)

# --------------------------------------------------

class TestMarginal(unittest.TestCase):
    def setUp(self):
        self.marginal_named = gr.MarginalNamed(
            d_name="norm", d_param={"loc": 0, "scale": 1}
        )

    def test_fcn(self):

        ## Invoke summary
        self.marginal_named.summary()

        ## Correct values for normal distribution
        self.assertTrue(self.marginal_named.l(0.5) == norm.pdf(0.5))
        self.assertTrue(self.marginal_named.p(0.5) == norm.cdf(0.5))
        self.assertTrue(self.marginal_named.q(0.5) == norm.ppf(0.5))
