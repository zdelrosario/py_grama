import unittest

from context import grama as gr
from context import data

from pandas import Series
from numpy import NaN, arange
from numpy.random import shuffle
from scipy.stats import norm

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

class TestPandasUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_fillna(self):
        s_nan = Series([NaN] * 2)
        s_filled = gr.fillna(s_nan, value=0)
        s_true = Series([0.0] * 2)

        self.assertTrue((s_filled == s_true).all())

class TestPareto(unittest.TestCase):
    def setUp(self):
        pass

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

    def test_stratum_min(self):
        df_test = gr.df_make(
            x=[1, 2, 0, 1, 2, 0, 1, 2],
            y=[0, 0, 1, 1, 1, 2, 2, 2],
            p=[1, 2, 1, 2, 3, 2, 3, 4],
        )

        # Test for accuracy
        self.assertTrue(
            (
                df_test
                >> gr.tf_mutate(p_comp=gr.stratum_min(X.x, X.y))
                >> gr.tf_mutate(flag=X.p == X.p_comp)
            ).flag.all()
        )

        # Check for ValueError
        with self.assertRaises(ValueError):
            gr.stratum_min([1], [1, 2, 3])

class TestQQ(unittest.TestCase):
    def setUp(self):
        pass

    def test_qqvals(self):
        # Set up data
        n = 10
        i = arange(1, n + 1)
        p = (i - 0.3175) / (len(i) + 0.365)
        p[0] = 1 - 0.5**(1/n)
        p[-1] = 0.5**(1/n)
        q = norm.ppf(p)

        # Correct values
        marg = gr.marg_mom("norm", mean=0, sd=1) # Use true distribution
        q_res = gr.qqvals(q, marg=marg)
        self.assertTrue((q == q_res).all())

        # Handles shuffling
        shuffle(q)
        self.assertTrue((q == gr.qqvals(q, marg=marg)).all())

class TestArray(unittest.TestCase):
    def setUp(self):
        pass

    def test_linspace(self):
        # Works in pipeline
        (
            gr.df_make(i=range(10))
            >> gr.tf_mutate(
                x=gr.linspace(0, 1, gr.n(X.index)),
                l=gr.logspace(0, 1, gr.n(X.index)),
            )
        )
