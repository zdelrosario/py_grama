import numpy as np
import pandas as pd
import unittest

from context import grama as gr
from context import data

##==============================================================================
## join test functions
##==============================================================================


class TestJoins(unittest.TestCase):
    def setUp(self):
        self.dfA = pd.DataFrame({"x1": ["A", "B", "C"], "x2": [1, 2, 3]})
        self.dfB = pd.DataFrame({"x1": ["A", "B", "D"], "x3": [True, False, True]})
        self.dfC = pd.DataFrame({"x1": ["B", "C", "D"], "x2": [2, 3, 4]})

    def test_inner_join(self):
        ab = pd.DataFrame({"x1": ["A", "B"], "x2": [1, 2], "x3": [True, False]})

        c = self.dfA >> gr.tf_inner_join(self.dfB, by="x1")
        self.assertTrue(c.equals(ab))

    def test_outer_join(self):
        ab = pd.DataFrame(
            {
                "x1": ["A", "B", "C", "D"],
                "x2": [1, 2, 3, np.nan],
                "x3": [True, False, np.nan, True],
            }
        )

        c = self.dfA >> gr.tf_outer_join(self.dfB, by="x1")
        self.assertTrue(c.equals(ab))
        c = self.dfA >> gr.tf_full_join(self.dfB, by="x1")
        self.assertTrue(c.equals(ab))

    def test_left_join(self):
        ab = pd.DataFrame(
            {"x1": ["A", "B", "C"], "x2": [1, 2, 3], "x3": [True, False, np.nan]}
        )

        c = self.dfA >> gr.tf_left_join(self.dfB, by="x1")
        self.assertTrue(c.equals(ab))

    def test_right_join(self):
        ab = pd.DataFrame(
            {"x1": ["A", "B", "D"], "x2": [1, 2, np.nan], "x3": [True, False, True]}
        )

        c = self.dfA >> gr.tf_right_join(self.dfB, by="x1")
        self.assertTrue(c.equals(ab))

    def test_semi_join(self):
        ab = pd.DataFrame({"x1": ["A", "B"], "x2": [1, 2]})

        c = self.dfA >> gr.tf_semi_join(self.dfB, by="x1")
        self.assertTrue(c.equals(ab))

    def test_anti_join(self):
        ab = pd.DataFrame({"x1": ["C"], "x2": [3]}, index=[2])

        c = self.dfA >> gr.tf_anti_join(self.dfB, by="x1")
        self.assertTrue(c.equals(ab))

    ##==============================================================================
    ## set operation (row join) test functions
    ##==============================================================================

    def test_union(self):
        ac = pd.DataFrame(
            {"x1": ["A", "B", "C", "D"], "x2": [1, 2, 3, 4]}, index=[0, 1, 2, 2]
        )

        d = self.dfA >> gr.tf_union(self.dfC)
        self.assertTrue(d.equals(ac))

    def test_intersect(self):
        ac = pd.DataFrame({"x1": ["B", "C"], "x2": [2, 3]})

        d = self.dfA >> gr.tf_intersect(self.dfC)
        self.assertTrue(d.equals(ac))

    def test_set_diff(self):
        ac = pd.DataFrame({"x1": ["A"], "x2": [1]})

        d = self.dfA >> gr.tf_set_diff(self.dfC)
        self.assertTrue(d.equals(ac))

    ##==============================================================================
    ## bind rows, cols
    ##==============================================================================

    def test_bind_rows(self):
        inner = pd.DataFrame({"x1": ["A", "B", "C", "A", "B", "D"]})
        outer = pd.DataFrame(
            {
                "x1": ["A", "B", "C", "A", "B", "D"],
                "x2": [1, 2, 3, np.nan, np.nan, np.nan],
                "x3": [np.nan, np.nan, np.nan, True, False, True],
            }
        )
        ab_inner = self.dfA >> gr.tf_bind_rows(self.dfB, join="inner")
        ab_outer = self.dfA >> gr.tf_bind_rows(self.dfB, join="outer")
        self.assertTrue(inner.equals(ab_inner.reset_index(drop=True)))
        self.assertTrue(outer.equals(ab_outer.reset_index(drop=True)))

    def test_bind_cols(self):
        dfB = self.dfB.copy()
        dfB.columns = ["x3", "x4"]
        df = pd.DataFrame(
            {
                "x1": ["A", "B", "C"],
                "x2": [1, 2, 3],
                "x3": ["A", "B", "D"],
                "x4": [True, False, True],
            }
        )
        d = self.dfA >> gr.tf_bind_cols(dfB)
        self.assertTrue(df.equals(d))
