import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import data

X = gr.Intention()

##==============================================================================
## filter helper tests
##==============================================================================


class TestSubset(unittest.TestCase):
    def test_dropna(self):
        df = gr.df_make(x=[1.0, 2.0, 3.0], y=[1.0, np.nan, 3.0], z=[1.0, 2.0, np.nan])

        df_true_default = gr.df_make(x=[1.0], y=[1.0], z=[1.0])
        df_true_y = gr.df_make(x=[1.0, 3.0], y=[1.0, 3.0], z=[1.0, np.nan])

        df_res_default = df >> gr.tf_dropna()
        df_res_y = df >> gr.tf_dropna(subset=["y"])

        self.assertTrue(gr.df_equal(df_true_default, df_res_default))
        self.assertTrue(gr.df_equal(df_true_y, df_res_y))
