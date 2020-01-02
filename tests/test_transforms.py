import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr
from context import data

## Test transform tools
##################################################
class TestTools(unittest.TestCase):

    def setUp(self):
        pass

    def test_bootstrap(self):
        df_stang = data.df_stang
        df_stang._meta = "foo"

        def tran_stats(df):
            val = df.select_dtypes(include="number").values

            means = np.mean(val, axis=0)
            stds = np.std(val, axis=0)

            # Check metadata propagation
            self.assertTrue(df._meta == "foo")

            return pd.DataFrame(
                data = {
                    "var": df.select_dtypes(include="number").columns,
                    "mean": means,
                    "std": stds
                }
            )

        df_res = gr.tran_bootstrap(
            df_stang,
            tran=tran_stats,
            n_boot=3e0,
            n_sub=3e0,
            seed=101
        )

        df_sel = gr.tran_bootstrap(
            df_stang,
            tran=tran_stats,
            n_boot=3e0,
            n_sub=3e0,
            seed=101,
            col_sel=["mean"]
        )

        df_piped = df_stang >> gr.tf_bootstrap(
            tran=tran_stats,
            n_boot=3e0,
            n_sub=3e0,
            seed=101
        )

        ## Test output shape
        self.assertTrue(
            set(df_res.columns) == set([
                "var", "mean", "mean_lo", "mean_hi", "std", "std_lo", "std_hi"
            ])
        )
        self.assertTrue(df_res.shape[0] == 4)

        self.assertTrue(
            set(df_sel.columns) == set([
                "var", "mean", "mean_lo", "mean_hi", "std"
            ])
        )
        self.assertTrue(df_sel.shape[0] == 4)

        ## Test pipe
        self.assertTrue(gr.df_equal(df_res, df_piped))

    def test_outer(self):
        df = pd.DataFrame(dict(x=[1,2]))
        df_outer = pd.DataFrame(dict(y=[3,4]))

        df_true = pd.DataFrame(dict(
            x=[1,2,1,2],
            y=[3,3,4,4]
        ))

        df_res = gr.tran_outer(df, df_outer)
        df_piped = df >> gr.tf_outer(df_outer)

        pd.testing.assert_frame_equal(
            df_true,
            df_res,
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )
        pd.testing.assert_frame_equal(
            df_piped,
            df_res,
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )

# --------------------------------------------------
class TestReshape(unittest.TestCase):
    def setUp(self):
        ## Transpose test
        self.df_rownames = pd.DataFrame({
            "rowname": ["a", "b"],
            "x":       [0, 1],
            "y":       [2, 3],
        })
        self.df_transposed = pd.DataFrame({
            "rowname": ["x", "y"],
            "a":       [ 0, 2],
            "b":       [ 1, 3]
        })

        ## Gather test
        self.df_wide = pd.DataFrame({
            "a": [0],
            "b": [1]
        })
        self.df_gathered = pd.DataFrame({
            "key": ["a", "b"],
            "value": [0, 1]
        })

        ## Spread test
        self.df_long = pd.DataFrame({
            "key":   ["a", "b"],
            "value": [  0,   1]
        })
        self.df_spreaded = pd.DataFrame({
            "index": ["value"], "a": [0], "b": [1]
        })
        self.df_spreaded_drop = pd.DataFrame({
            "a": [0], "b": [1]
        })

    def test_gather(self):
        df_res = gr.tran_gather(self.df_wide, "key", "value", ["a", "b"])
        self.assertTrue(self.df_gathered.equals(df_res))

        ## Test pipe
        df_piped = self.df_wide >> gr.tf_gather("key", "value", ["a", "b"])
        self.assertTrue(df_res.equals(df_piped))

    def test_spread(self):
        df_res = gr.tran_spread(self.df_long, "key", "value")
        self.assertTrue(self.df_spreaded.equals(df_res))

        ## Test pipe
        df_piped = self.df_long >> gr.tf_spread("key", "value")
        self.assertTrue(df_res.equals(df_piped))

        ## Test with drop
        self.assertTrue(
            self.df_spreaded_drop.equals(
                gr.tran_spread(self.df_long, "key", "value", drop=True)
            )
        )
