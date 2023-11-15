import numpy as np
import pandas as pd
from scipy.stats import norm, beta, lognorm, weibull_min
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

        d_gkde = mg_gkde.d(np.array([1, 10000, 10400, 10800, 1e6]))
        p_gkde = mg_gkde.p(np.array([1, 10000, 10400, 10800, 1e6]))
        q_gkde = mg_gkde.q(np.array([0.0, 0.25, 0.50, 0.75, 1.0]))
        mg_gkde.summary()

        self.assertTrue(np.isclose(q_gkde[2], self.median, atol=0, rtol=0.05))

        # Test random variable sample
        n = 10
        n_gkde = mg_gkde.r(n)
        self.assertTrue(len(n_gkde) == n)

        # Copy
        mg_new = mg_gkde.copy()
        self.assertTrue(all(np.isclose(
            mg_new.d(q_gkde),
            mg_gkde.d(q_gkde),
        )))
        self.assertTrue(all(np.isclose(
            mg_new.p(q_gkde),
            mg_gkde.p(q_gkde),
        )))

    def test_fit(self):
        mg_norm = gr.marg_fit("norm", data.df_stang.E)

        d_norm = mg_norm.d(np.array([10000, 10400, 10800]))
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
            gr.marg_fit("norm", data.df_stang)
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

        ## Test 2-parameter lognormal
        mg_log = gr.marg_mom(
            "lognorm",
            mean=5e4,
            var=5e2**2,
            floc=0,
        )

        self.assertTrue(all(np.isclose(
            lognorm(**mg_log.d_param).stats("mv"),
            np.array([5e4, 5e2**2])
        )))

        ## Test 2-parameter Weibull fit
        mg_weibull2 = gr.marg_mom(
            "weibull_min",
            mean=5e4,
            var=5e2**2,
            floc=0,
        )

        self.assertTrue(all(np.isclose(
            weibull_min(**mg_weibull2.d_param).stats("mv"),
            np.array([5e4, 5e2**2])
        )))

        ## Test bounded uniform
        mg_unif_bnd = gr.marg_mom("uniform", lo=-1, up=+1)
        self.assertTrue(all(np.isclose(
            (mg_unif_bnd.d_param["loc"], mg_unif_bnd.d_param["scale"]),
            (-1, +2)
        )))

        ## Test bounded beta
        mg_beta_bnd1 = gr.marg_mom("beta", lo=-1, up=+1, mean=0, var=1/8)
        self.assertTrue(all(np.isclose(
            beta(**mg_beta_bnd1.d_param).stats("mv"),
            np.array([0, 1/8])
        )))
        mg_beta_bnd2 = gr.marg_mom("beta", lo=-1, up=+1, mean=0.2, var=1/8)
        self.assertTrue(all(np.isclose(
            beta(**mg_beta_bnd2.d_param).stats("mv"),
            np.array([0.2, 1/8])
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
        # Cannot use lo or up with non-bounded dist
        with self.assertRaises(ValueError):
            gr.marg_mom("lognorm", mean=1, sd=1, kurt=1, kurt_excess=-2, lo=0)
        with self.assertRaises(ValueError):
            gr.marg_mom("lognorm", mean=1, sd=1, kurt=1, kurt_excess=-2, up=0)

    def test_trunc(self):
        mg_base = gr.marg_mom("norm", mean=0, sd=1)

        ## Two-sided truncation
        mg_trunc1 = gr.marg_trunc(mg_base, lo=-2, up=+2)
        # Summary
        mg_trunc1.summary()

        # Quantiles
        self.assertTrue(abs(mg_trunc1.q(0) + 2) < 1e-6)
        self.assertTrue(abs(mg_trunc1.q(1) - 2) < 1e-6)
        # PDF truncated
        self.assertTrue(abs(mg_trunc1.d(-2.1)) < 1e-6)
        self.assertTrue(abs(mg_trunc1.d(+2.1)) < 1e-6)
        # CDF truncated
        self.assertTrue(abs(mg_trunc1.p(-2.1) - 0) < 1e-6)
        self.assertTrue(abs(mg_trunc1.p(+2.1) - 1) < 1e-6)

        # Vector evaluation
        qv = mg_trunc1.q(np.array([0, 0.5, 1]))
        self.assertTrue(all(np.isclose(qv, np.array([-2, 0, +2]))))
        dv = mg_trunc1.d(np.array([-2.1, 0, +2.1]))
        self.assertTrue(all(np.isclose(
            dv,
            np.array([0, mg_base.d(0) / (mg_base.p(2) - mg_base.p(-2)), 0])
        )))
        pv = mg_trunc1.p(np.array([-2.1, 0, +2.1]))
        self.assertTrue(all(np.isclose(pv, np.array([0, 0.5, 1]))))

        ## Lower truncation
        mg_trunc_lo = gr.marg_trunc(mg_base, lo=0)
        # Quantiles
        self.assertTrue(abs(mg_trunc_lo.q(0) - 0) < 1e-6)
        self.assertTrue(np.isinf(mg_trunc_lo.q(1)))
        # PDF truncated
        self.assertTrue(abs(mg_trunc_lo.d(-0.1)) < 1e-6)
        # PDF doubled for truncation at center
        self.assertTrue(abs(mg_trunc_lo.d(0) - mg_base.d(0) * 2) < 1e-6)
        # CDF truncated
        self.assertTrue(abs(mg_trunc_lo.p(-0.1) - 0) < 1e-6)
        self.assertTrue(abs(mg_trunc_lo.p(+np.Inf) - 1) < 1e-6)

        ## Upper truncation
        mg_trunc_up = gr.marg_trunc(mg_base, up=0)
        # Quantiles
        self.assertTrue(np.isinf(mg_trunc_up.q(0)))
        self.assertTrue(abs(mg_trunc_up.q(1) - 0) < 1e-6)
        # PDF truncated
        self.assertTrue(abs(mg_trunc_up.d(+0.1)) < 1e-6)
        # PDF doubled for truncation at center
        self.assertTrue(abs(mg_trunc_up.d(0) - mg_base.d(0) * 2) < 1e-6)
        # CDF truncated
        self.assertTrue(abs(mg_trunc_up.p(-np.Inf) - 0) < 1e-6)
        self.assertTrue(abs(mg_trunc_up.p(+0.1) - 1) < 1e-6)

        # Copy
        mg_copy = mg_trunc1.copy()
        self.assertTrue(all(np.isclose(
            mg_copy.q(np.array([0, 0.5, 1])),
            mg_trunc1.q(np.array([0, 0.5, 1])),
        )))

        # Invalid truncation
        with self.assertRaises(ValueError):
            gr.marg_trunc(mg_base)

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
        self.assertTrue(self.marginal_named.d(0.5) == norm.pdf(0.5))
        self.assertTrue(self.marginal_named.p(0.5) == norm.cdf(0.5))
        self.assertTrue(self.marginal_named.q(0.5) == norm.ppf(0.5))
