import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import models, data

## Core function tests
##################################################
class TestPipe(unittest.TestCase):
    def setUp(self):
        self.md = models.make_test()

    def test_pipe(self):
        ## Chain
        res = self.md >> gr.ev_hybrid(df_det="nom") >> gr.tf_sobol()


class TestMarginals(unittest.TestCase):
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


class TestMisc(unittest.TestCase):
    def setUp(self):
        pass

    def test_df_equal(self):
        df1 = pd.DataFrame(dict(x=[0], y=[0]))
        df2 = pd.DataFrame(dict(x=[0]))

        self.assertTrue(gr.df_equal(df1, df1))
        self.assertTrue(gr.df_equal(df1, df2) == False)

    def test_df_make(self):
        df_true = pd.DataFrame(dict(x=[0, 1], y=[0, 0], z=[1, 1]))
        df_res = gr.df_make(x=[0, 1], y=[0], z=1)

        with self.assertRaises(ValueError):
            gr.df_make(x=[1, 2, 3], y=[1, 2])
