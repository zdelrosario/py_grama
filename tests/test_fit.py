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
        ## Regression model
        self.md_true = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: [x, x + 1], var=["x"], out=["y", "z"])
            >> gr.cp_marginals(x={"dist": "uniform", "loc": 0, "scale": 2})
            >> gr.cp_copula_independence()
        )

        self.df = pd.DataFrame(dict(x=[0, 1, 2], y=[0, 1, 2], z=[1, 1, 1]))
        self.inputs = ["x"]
        self.outputs = ["y", "z"]

        ## Cluster model
        self.df_cluster = pd.DataFrame(
            dict(
                x=[0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
                y=[0.3, 0.2, 0.1, 1.3, 1.2, 1.1],
                c=[0, 0, 0, 1, 1, 1],
            )
        )

    def test_gp(self):
        ## Fit routine creates usable model
        md_fit = fit.fit_gp(self.df, md=self.md_true)
        df_res = gr.eval_df(md_fit, self.df[self.inputs])

        ## GP provides std estimates
        # self.assertTrue("y_std" in df_res.columns)

        ## GP is an interpolation
        self.assertTrue(gr.df_equal(df_res, self.df, close=True))

        ## Fit copies model data
        self.assertTrue(set(self.outputs) == set(self.md_true.out))

    def test_rf(self):
        ## Fit routine creates usable model
        md_fit = fit.fit_rf(self.df, md=self.md_true,)
        df_res = gr.eval_df(md_fit, self.df[self.inputs])

        ## How to test accuracy?
        ## TODO

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
