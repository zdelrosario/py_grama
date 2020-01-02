import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import models

## Core function tests
##################################################
class TestPipe(unittest.TestCase):
    def setUp(self):
        self.md = models.make_test()

    def test_pipe(self):
        ## Chain
        res = self.md >> \
              gr.ev_hybrid(df_det="nom") >> \
              gr.tf_sobol()

class TestMisc(unittest.TestCase):
    def setUp(self):
        pass

    def test_df_equal(self):
        df1 = pd.DataFrame(dict(x=[0], y=[0]))
        df2 = pd.DataFrame(dict(x=[0]))

        self.assertTrue(gr.df_equal(df1, df1))
        self.assertTrue(gr.df_equal(df1, df2) == False)
