import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import fit

## Core function tests
##################################################
class TestFits(unittest.TestCase):
    """Test implementations of fitting procedures
    """

    def setUp(self):
        self.df = pd.DataFrame(dict(x=[0, 1, 2], y=[0, 1, 2], z=[1, 1, 1]))
        self.inputs = ["x"]
        self.outputs = ["y", "z"]

    def test_gp(self):
        md_true = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: [x, x + 1], var=["x"], out=["y", "z"])
            >> gr.cp_marginals(x={"dist": "uniform", "loc": 0, "scale": 2})
        )

        md_fit = fit.fit_gp(self.df, md=md_true)
        df_res = gr.eval_df(md_fit, self.df[self.inputs])

        ## GP provides std estimates
        # self.assertTrue("y_std" in df_res.columns)

        ## GP is an interpolation
        self.assertTrue(gr.df_equal(df_res, self.df, close=True))

        ## Fit copies model data
        self.assertTrue(set(self.outputs) == set(md_true.out))

    def test_rf(self):
        pass
