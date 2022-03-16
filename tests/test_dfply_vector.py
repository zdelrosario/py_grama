import numpy as np
import pandas as pd
import unittest

from context import grama as gr
from context import data

X = gr.Intention()


class TestVector(unittest.TestCase):
    ##==============================================================================
    ## desc, order by tests
    ##==============================================================================

    def test_desc(self):
        df = data.df_diamonds >> gr.tf_select(X.cut, X.x) >> gr.tf_head(10)
        t = df >> gr.tf_summarize(
            last=gr.nth(X.x, -1, order_by=[gr.desc(X.cut), gr.desc(X.x)])
        )

        series_num = pd.Series([4, 1, 3, 2])
        series_bool = pd.Series([True, False, True, False])
        series_str = pd.Series(["d", "a", "c", "b"])

        num_truth = series_num.rank(method="min", ascending=False)
        bool_truth = series_bool.rank(method="min", ascending=False)
        str_truth = series_str.rank(method="min", ascending=False)

        self.assertTrue(gr.desc(series_num).equals(num_truth))
        self.assertTrue(gr.desc(series_bool).equals(bool_truth))
        self.assertTrue(gr.desc(series_str).equals(str_truth))

    def test_order_series_by(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
        order1 = pd.Series(["A", "B", "A", "B", "A", "B", "A", "B"])
        ordered1 = gr.order_series_by(series, order1).reset_index(drop=True)
        true1 = pd.Series([1, 3, 5, 7, 2, 4, 6, 8])
        self.assertTrue(ordered1.equals(true1))

        order2 = pd.Series([2, 2, 2, 2, 1, 1, 1, 1])
        ordered2 = gr.order_series_by(series, [order1, order2]).reset_index(drop=True)
        true2 = pd.Series([5, 7, 1, 3, 6, 8, 2, 4])
        self.assertTrue(ordered2.equals(true2))

    ##==============================================================================
    ## coalesce test
    ##==============================================================================

    def test_coalesce(self):
        df = pd.DataFrame(
            {
                "a": [1, np.nan, np.nan, np.nan, np.nan],
                "b": [2, 3, np.nan, np.nan, np.nan],
                "c": [np.nan, np.nan, 4, 5, np.nan],
                "d": [6, 7, 8, 9, np.nan],
            }
        )
        truth_df = df.assign(coal=[1, 3, 4, 5, np.nan])
        d = df >> gr.tf_mutate(coal=gr.coalesce(X.a, X.b, X.c, X.d))
        self.assertTrue(truth_df.equals(d))

    ##==============================================================================
    ## case_when test
    ##==============================================================================

    def test_case_when(self):
        df = pd.DataFrame({"num": np.arange(31)})
        df_truth = df.assign(
            strnum=[
                "fizzbuzz"
                if (i % 15 == 0)
                else "fizz"
                if (i % 3 == 0)
                else "buzz"
                if (i % 5 == 0)
                else str(i)
                for i in np.arange(31)
            ]
        )
        d = df >> gr.tf_mutate(
            strnum=gr.case_when(
                [X.num % 15 == 0, "fizzbuzz"],
                [X.num % 3 == 0, "fizz"],
                [X.num % 5 == 0, "buzz"],
                [True, X.num.astype(str)],
            )
        )
        self.assertTrue(df_truth.equals(d))

    ##==============================================================================
    ## if_else test
    ##==============================================================================

    def test_if_else(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        b_truth = ["odd", "even", "odd", "even", "odd", "even", "odd", "even", "odd"]
        d = df >> gr.tf_mutate(b=gr.if_else(X.a % 2 == 0, "even", "odd"))
        self.assertTrue(d.equals(df.assign(b=b_truth)))

        df = pd.DataFrame({"a": [0, 0, 0, 1, 1, 1, 2, 2, 2]})
        b_truth = [5, 5, 5, 5, 5, 5, 9, 9, 9]
        d = df >> gr.tf_mutate(
            b=gr.if_else(
                X.a < 2, [5, 5, 5, 5, 5, 5, 5, 5, 5], [9, 9, 9, 9, 9, 9, 9, 9, 9]
            )
        )
        self.assertTrue(d.equals(df.assign(b=b_truth)))

    ##==============================================================================
    ## na_if test
    ##==============================================================================

    def test_na_if(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        d = df >> gr.tf_mutate(b=gr.na_if(X.a, 3), c=gr.na_if(X.a, 1, 2, 3))
        d = d[["a", "b", "c"]]
        df_true = df.assign(b=[1, 2, np.nan, 4, 5], c=[np.nan, np.nan, np.nan, 4, 5])
        self.assertTrue(df_true.equals(d))
