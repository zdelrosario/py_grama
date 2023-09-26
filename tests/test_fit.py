import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import models
from context import data

X = gr.Intention()

## Core function tests
##################################################
class TestFits(unittest.TestCase):
    """Test implementations of fitting procedures"""

    def setUp(self):
        ## Smooth model
        self.md_smooth = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: [x, x + 1], var=["x"], out=["y", "z"])
            # >> gr.cp_vec_function(fun=lambda df: gr.df_make(y=df.x, z=df.x + 1), var=["x"], out=["y", "z"])
            >> gr.cp_marginals(x={"dist": "uniform", "loc": 0, "scale": 2})
            >> gr.cp_copula_independence()
        )
        self.df_smooth = self.md_smooth >> gr.ev_df(df=pd.DataFrame(dict(x=[0, 1, 2])))

        ## Tree model
        self.md_tree = (
            gr.Model()
            # >> gr.cp_function(fun=lambda x: [0, x < 5], var=["x"], out=["y", "z"])
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(y=df.x, z=df.x + 1), var=["x"], out=["y", "z"]
            )
            >> gr.cp_marginals(x={"dist": "uniform", "loc": 0, "scale": 2})
            >> gr.cp_copula_independence()
        )

        self.df_tree = self.md_tree >> gr.ev_df(
            df=pd.DataFrame(dict(x=np.linspace(0, 10, num=8)))
        )

        ## Cluster model
        self.df_cluster = pd.DataFrame(
            dict(
                x=[0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4],
                y=[0.3, 0.2, 0.1, 0.0, 1.3, 1.2, 1.1, 1.0],
                c=[0, 0, 0, 0, 1, 1, 1, 1],
            )
        )

    def test_gp(self):
        ## Fit routine creates usable model
        md_fit = gr.fit_gp(self.df_smooth, md=self.md_smooth)
        df_res = gr.eval_df(md_fit, self.df_smooth[self.md_smooth.var])

        ## GP provides std estimates
        self.assertTrue("y_sd" in df_res.columns)

        ## GP is an interpolation
        self.assertTrue(
            gr.df_equal(
                df_res[["x", "y_mean", "z_mean"]].rename(
                    {"y_mean": "y", "z_mean": "z"}, axis=1
                ),
                self.df_smooth,
                close=True,
            )
        )

        ## Fit copies model data
        self.assertTrue(set(md_fit.var) == set(self.md_smooth.var))
        self.assertTrue(
            set(md_fit.out)
            == set(map(lambda s: s + "_mean", self.md_smooth.out)).union(
                set(map(lambda s: s + "_sd", self.md_smooth.out))
            )
        )

    def test_rf(self):
        ## Fit routine creates usable model
        md_fit = gr.fit_rf(
            self.df_tree,
            md=self.md_tree,
            max_depth=1,  # True tree is a stump
            seed=101,
        )
        df_res = gr.eval_df(md_fit, self.df_tree[self.md_tree.var])

        ## RF can approximately recover a tree; check ends only
        self.assertTrue(
            gr.df_equal(
                df_res[["y_mean", "z_mean"]].iloc[[0, 1, -2, -1]],
                self.df_tree[["y", "z"]].iloc[[0, 1, -2, -1]]
                >> gr.tf_rename(y_mean="y", z_mean="z"),
                close=True,
                precision=1,
            )
        )

        ## Fit copies model data
        self.assertTrue(set(md_fit.var) == set(self.md_tree.var))
        self.assertTrue(
            set(md_fit.out) == set(map(lambda s: s + "_mean", self.md_tree.out))
        )

    def test_lm(self):
        ## Fit routine creates usable model
        md_fit = gr.fit_lm(
            self.df_smooth,
            md=self.md_smooth,
        )
        df_res = gr.eval_df(md_fit, self.df_smooth[self.md_smooth.var])

        ## LM can recover a linear model
        self.assertTrue(
            gr.df_equal(
                df_res,
                self.df_smooth >> gr.tf_rename(y_mean="y", z_mean="z"),
                close=True,
                precision=1,
            )
        )

        ## Fit copies model data
        self.assertTrue(set(md_fit.var) == set(self.md_smooth.var))
        self.assertTrue(
            set(md_fit.out) == set(map(lambda s: s + "_mean", self.md_smooth.out))
        )

    def test_kmeans(self):
        ## Fit routine creates usable model
        var = ["x", "y"]
        md_fit = gr.fit_kmeans(self.df_cluster, var=var, n_clusters=2)
        df_res = gr.eval_df(md_fit, self.df_cluster[var])

        ## Check correctness
        # Match clusters by min(x)
        id_true = (self.df_cluster >> gr.tf_filter(X.x == gr.min(X.x))).c[0]
        id_res = (df_res >> gr.tf_filter(X.x == gr.min(X.x))).cluster_id[0]

        df_res1 = (
            self.df_cluster >> gr.tf_filter(X.c == id_true) >> gr.tf_select(X.x, X.y)
        )
        df_res2 = (
            df_res >> gr.tf_filter(X.cluster_id == id_res) >> gr.tf_select(X.x, X.y)
        )

        self.assertTrue(gr.df_equal(df_res1, df_res2))

    def test_nls(self):
        ## Ground-truth model
        c_true = 2
        a_true = 1

        md_true = (
            gr.Model()
            >> gr.cp_function(
                fun=lambda x, epsilon: a_true * np.exp(x * c_true) + epsilon,
                var=["x", "epsilon"],
                out=["y"],
            )
            >> gr.cp_marginals(epsilon={"dist": "norm", "loc": 0, "scale": 0.5})
            >> gr.cp_copula_independence()
        )
        df_data = md_true >> gr.ev_sample(
            n=5, seed=101, df_det=gr.df_make(x=[0, 1, 2, 3, 4])
        )

        ## Model to fit
        md_param = (
            gr.Model()
            >> gr.cp_function(
                fun=lambda x, c, a: a * np.exp(x * c),
                var=["x", "c", "a"],
                out=["y"]
            )
            >> gr.cp_bounds(c=[0, 4], a=[0.1, 2.0])
        )

        ## Fit the model
        md_fit = df_data >> gr.ft_nls(
            md=md_param,
            verbose=False,
            uq_method="linpool",
        )

        ## Unidentifiable model throws warning
        # -------------------------
        md_unidet = (
            gr.Model()
            >> gr.cp_function(
                fun=lambda x, c, a, z: a / z * np.exp(x * c),
                var=["x", "c", "a", "z"],
                out=["y"],
            )
            >> gr.cp_bounds(c=[0, 4], a=[0.1, 2.0], z=[0, 1])
        )
        with self.assertWarns(RuntimeWarning):
            gr.fit_nls(
                df_data,
                md=md_unidet,
                uq_method="linpool",
            )

        ## True parameters in wide confidence region
        # -------------------------
        alpha = 1e-3
        self.assertTrue(
            (md_fit.density.marginals["c"].q(alpha / 2) <= c_true)
            and (c_true <= md_fit.density.marginals["c"].q(1 - alpha / 2))
        )

        self.assertTrue(
            (md_fit.density.marginals["a"].q(alpha / 2) <= a_true)
            and (a_true <= md_fit.density.marginals["a"].q(1 - alpha / 2))
        )

        ## Model with fixed parameter
        # -------------------------
        md_fixed = (
            gr.Model()
            >> gr.cp_function(
                fun=lambda x, c, a, e: a * np.exp(x * c) + e,
                var=["x", "c", "a", "e"],
                out=["y"],
            )
            >> gr.cp_bounds(c=[0, 4], a=[1, 1], e=[0, 0])
        )
        md_fit_fixed = df_data >> gr.ft_nls(
            md=md_fixed, verbose=False, uq_method="linpool"
        )

        # Test that fixed model can evaluate successfully
        gr.eval_sample(md_fit_fixed, n=1, df_det="nom")

        ## Model with fixed parameter and two inputs
        # -------------------------
        df_data2 = (
            gr.df_grid(x=(1,2,3,4), y=(1,2,3,4))
            >> gr.tf_mutate(f=2*X.x + X.y**2)
        )
        md_fixed2 = (
            gr.Model()
            >> gr.cp_function(
                fun=lambda x, y, a, b: a*x + b*y**2,
                var=["x", "y", "a", "b"],
                out=["f"],
            )
            >> gr.cp_bounds(a=(0, 3), b=(1,1))
        )
        md_fit_fixed2 = df_data2 >> gr.ft_nls(
            md=md_fixed2, verbose=False, uq_method="linpool"
        )

        # Test that fixed model can evaluate successfully
        gr.eval_sample(md_fit_fixed2, n=1, df_det="nom")

        ## Trajectory model
        # -------------------------
        md_base = models.make_trajectory_linear()
        md_fit = data.df_trajectory_windowed >> gr.ft_nls(
            md=md_base, method="SLSQP", tol=1e-3
        )
        df_tmp = md_fit >> gr.ev_nominal(df_det="nom")

        ## Select output for fitting
        # -------------------------
        # Split model has inconsistent "true" parameter value
        md_split = (
            gr.Model("Split")
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(
                    f=1 * df.c * df.x,
                    g=2 * df.c * df.x,
                ),
                var=["c", "x"],
                out=["f", "g"],
            )
            >> gr.cp_bounds(
                x=(-1, +1),
                c=(-1, +1),
            )
        )

        df_split = gr.df_make(x=gr.linspace(-1, +1, 100)) >> gr.tf_mutate(f=X.x, g=X.x)

        # Fitting both outputs: cannot achieve mse ~= 0
        df_both = (
            df_split
            >> gr.ft_nls(md_split, out=["f", "g"])
            >> gr.ev_df(df_split >> gr.tf_rename(f_t=X.f, g_t=X.g))
            >> gr.tf_summarize(
                mse_f=gr.mse(X.f, X.f_t),
                mse_g=gr.mse(X.g, X.g_t),
            )
        )
        self.assertTrue(df_both.mse_f[0] > 0)
        self.assertTrue(df_both.mse_g[0] > 0)

        # Fitting "f" only
        df_f = (
            df_split
            >> gr.ft_nls(md_split, out=["f"])
            >> gr.ev_df(df_split >> gr.tf_rename(f_t=X.f, g_t=X.g))
            >> gr.tf_summarize(
                mse_f=gr.mse(X.f, X.f_t),
                mse_g=gr.mse(X.g, X.g_t),
            )
        )
        self.assertTrue(df_f.mse_f[0] < 1e-16)
        self.assertTrue(df_f.mse_g[0] > 0)

        # Fitting "g" only
        df_g = (
            df_split
            >> gr.ft_nls(md_split, out=["g"])
            >> gr.ev_df(df_split >> gr.tf_rename(f_t=X.f, g_t=X.g))
            >> gr.tf_summarize(
                mse_f=gr.mse(X.f, X.f_t),
                mse_g=gr.mse(X.g, X.g_t),
            )
        )
        self.assertTrue(df_g.mse_f[0] > 0)
        self.assertTrue(df_g.mse_g[0] < 1e-16)
