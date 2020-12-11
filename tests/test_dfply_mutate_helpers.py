import unittest

from context import grama as gr
from context import data

from pandas import Series
from numpy import NaN

X = gr.Intention()

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

    def test_pareto_min(self):
        df_test = gr.df_make(
            x=[1, 2, 0, 1, 2, 0, 1, 2],
            y=[0, 0, 1, 1, 1, 2, 2, 2],
            p=[1, 0, 1, 0, 0, 0, 0, 0],
        )

        # Test for accuracy
        self.assertTrue(
            (
                df_test
                >> gr.tf_mutate(p_comp=gr.pareto_min(X.x, X.y))
                >> gr.tf_mutate(flag=X.p == X.p_comp)
            ).flag.all()
        )

        # Check for ValueError
        with self.assertRaises(ValueError):
            gr.pareto_min([1], [1, 2, 3])
