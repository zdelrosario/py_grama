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


class TestHelpers(unittest.TestCase):
    def test_var_in(self):
        df = data.df_diamonds[["cut"]].head(10)
        d1 = df[(df.cut == "Ideal") | (df.cut == "Premium")].reset_index(drop=True)
        d2 = df >> gr.tf_filter(gr.var_in(X.cut, ["Ideal", "Premium"]))

        self.assertTrue(d1.equals(d2))
