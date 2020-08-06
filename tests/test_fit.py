import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import fit

X = gr.Intention()

## Core function tests
##################################################
class TestFits(unittest.TestCase):
    """Test implementations of fitting procedures
    """

    def setUp(self):
        ## Smooth model
        self.md_smooth = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: [x, x + 1], var=["x"], out=["y", "z"])
            >> gr.cp_marginals(x={"dist": "uniform", "loc": 0, "scale": 2})
            >> gr.cp_copula_independence()
        )

        self.df_smooth = self.md_smooth >> gr.ev_df(df=pd.DataFrame(dict(x=[0, 1, 2])))

        ## Tree model
        self.md_tree = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: [0, x < 5], var=["x"], out=["y", "z"])
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
        md_fit = fit.fit_gp(self.df_smooth, md=self.md_smooth)
        df_res = gr.eval_df(md_fit, self.df_smooth[self.md_smooth.var])

        ## GP provides std estimates
        # self.assertTrue("y_std" in df_res.columns)

        ## GP is an interpolation
        self.assertTrue(gr.df_equal(df_res, self.df_smooth, close=True))

        ## Fit copies model data
        self.assertTrue(set(md_fit.var) == set(self.md_smooth.var))
        self.assertTrue(set(md_fit.out) == set(self.md_smooth.out))

    def test_rf(self):
        ## Fit routine creates usable model
        md_fit = fit.fit_rf(
            self.df_tree,
            md=self.md_tree,
            max_depth=1,  # True tree is a stump
            seed=101,
        )
        df_res = gr.eval_df(md_fit, self.df_tree[self.md_tree.var])

        ## RF can approximately recover a tree
        self.assertTrue(gr.df_equal(df_res, self.df_tree, close=True, precision=1,))

        ## Fit copies model data
        self.assertTrue(set(md_fit.var) == set(self.md_tree.var))
        self.assertTrue(set(md_fit.out) == set(self.md_tree.out))

    def test_lolo(self):
        ## Fit routine creates usable model
        md_fit = fit.fit_lolo(
            self.df_tree,
            md=self.md_tree,
            max_depth=1,  # True tree is a stump
            seed=102,
        )
        df_res = gr.eval_df(md_fit, self.df_tree[self.md_tree.var])

        ## lolo seems to interpolate middle values; check ends only
        self.assertTrue(
            gr.df_equal(
                df_res[["y", "z"]].iloc[[0, 1, -2, -1]],
                self.df_tree[["y", "z"]].iloc[[0, 1, -2, -1]],
                close=True,
                precision=1,
            )
        )

        ## Fit copies model data, plus predictive std
        self.assertTrue(set(md_fit.var) == set(self.md_tree.var))
        self.assertTrue(
            set(md_fit.out)
            == set(self.md_tree.out + list(map(lambda s: s + "_std", self.md_tree.out)))
        )

    def test_kmeans(self):
        ## Fit routine creates usable model
        var = ["x", "y"]
        md_fit = fit.fit_kmeans(self.df_cluster, var=var, n_clusters=2)
        df_res = gr.eval_df(md_fit, self.df_cluster[var])

        ## Check correctness
        # Match clusters by min(x)
        id_true = (self.df_cluster >> gr.tf_filter(X.x == gr.colmin(X.x))).c[0]
        id_res = (df_res >> gr.tf_filter(X.x == gr.colmin(X.x))).cluster_id[0]

        df_res1 = (
            self.df_cluster >> gr.tf_filter(X.c == id_true) >> gr.tf_select(X.x, X.y)
        )
        df_res2 = (
            df_res >> gr.tf_filter(X.cluster_id == id_res) >> gr.tf_select(X.x, X.y)
        )

        self.assertTrue(gr.df_equal(df_res1, df_res2))
