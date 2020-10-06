import unittest

from context import grama as gr
from context import data

from pandas import Series
from numpy import NaN

##==============================================================================
## mask helper tests
##==============================================================================


class TestFactors(unittest.TestCase):
    def setUp(self):
        pass

    def test_fct_reorder(self):
        ang_fct = gr.fct_reorder(data.df_stang.ang, data.df_stang.E)

        self.assertTrue(list(ang_fct.categories) == [0, 90, 45])

    def test_fillna(self):
        s_nan = Series([NaN] * 2)
        s_filled = gr.fillna(s_nan, value=0)
        s_true = Series([0.0] * 2)

        self.assertTrue((s_filled == s_true).all())
