import numpy as np
import pandas as pd
from scipy.stats import norm, beta
import unittest

from context import grama as gr
from context import models, data

## Marginals function tests
##################################################
class TestMarginalTools(unittest.TestCase):
    def setUp(self):
        self.median = np.median(data.df_stang.E)

    def test_gkde(self):
        mg_gkde = gr.marg_gkde(data.df_stang.E)

        l_gkde = mg_gkde.l(np.array([1, 10000, 10400, 10800, 1e6]))
        p_gkde = mg_gkde.p(np.array([1, 10000, 10400, 10800, 1e6]))
        q_gkde = mg_gkde.q(np.array([0.0, 0.25, 0.50, 0.75, 1.0]))
        mg_gkde.summary()

        self.assertTrue(np.isclose(q_gkde[2], self.median, atol=0, rtol=0.05))

        # Test random variable sample
        n = 10
        n_gkde = mg_gkde.r(n)
        self.assertTrue(len(n_gkde) == n)

    def test_named(self):
        mg_norm = gr.marg_named(data.df_stang.E, "norm")

        l_norm = mg_norm.l(np.array([10000, 10400, 10800]))
        p_norm = mg_norm.p(np.array([10000, 10400, 10800]))
        q_norm = mg_norm.q(np.array([0.25, 0.50, 0.75]))
        mg_norm.summary()

        self.assertTrue(np.isclose(q_norm[1], self.median, atol=0, rtol=0.05))

        # Test random variable sample
        n = 10
        n_norm = mg_norm.r(n)
        self.assertTrue(len(n_norm) == n)

        ## Raises error when dataframe passed
        with self.assertRaises(ValueError):
            gr.marg_named(data.df_stang, "norm")
        with self.assertRaises(ValueError):
            gr.marg_gkde(data.df_stang)

    def test_mom(self):
        ## Test for accuracy
        mg_mom = gr.marg_mom(
            "beta",
            mean=1,
            sd=1,
            skew=0,
            kurt=2,
        )

        self.assertTrue(all(np.isclose(
            beta(**mg_mom.d_param).stats("mvsk"),
            np.array([1, 1, 0, 2 - 3])
        )))

        ## Test COV specification
        mg_cov = gr.marg_mom(
            "beta",
            mean=1,
            cov=2,
            skew=0,
            kurt=2,
        )

        self.assertTrue(all(np.isclose(
            beta(**mg_cov.d_param).stats("mvsk"),
            np.array([1, 4, 0, 2 - 3])
        )))

        ## Test invariants
        # Must provide mean
        with self.assertRaises(ValueError):
            gr.marg_mom("norm", sd=1)
        # Must provide sd or var
        with self.assertRaises(ValueError):
            gr.marg_mom("norm", mean=1)
        # Must provide sufficient parameters
        with self.assertRaises(ValueError):
            gr.marg_mom("lognorm", mean=1, sd=1)
        # Must not overdetermine
        with self.assertRaises(ValueError):
            gr.marg_mom("norm", mean=1, sd=1, skew=0, kurt=3)
        # For beta; skew == 0 and kurt == 4 is infeasible
        with self.assertRaises(RuntimeError):
            gr.marg_mom("beta", mean=1, sd=1, skew=0, kurt=4)
        # Cannot provide more than one of `sd`, `cov`, or `var`
        with self.assertRaises(ValueError):
            gr.marg_mom("norm", mean=1, sd=1, var=1)
        with self.assertRaises(ValueError):
            gr.marg_mom("norm", mean=1, sd=1, cov=1)
        with self.assertRaises(ValueError):
            gr.marg_mom("norm", mean=1, cov=1, var=1)
        with self.assertRaises(ValueError):
            gr.marg_mom("norm", mean=1, sd=1, var=1, cov=1)
        # Cannot provide both kurt and kurt_excess
        with self.assertRaises(ValueError):
            gr.marg_mom("lognorm", mean=1, sd=1, kurt=1, kurt_excess=-2)

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
