import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr
from context import data

## Test the built-in datasets
##################################################
class TestTools(unittest.TestCase):

    def setUp(self):
        pass

    def test_bootstrap(self):
        df_stang = data.df_stang

        def tran_stats(df):
            val = df.select_dtypes(include="number").values

            means = np.mean(val, axis=0)
            stds = np.std(val, axis=0)

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
            n_boot=3.1,
            n_sub=3.1,
            seed=101
        )

        df_sel = gr.tran_bootstrap(
            df_stang,
            tran=tran_stats,
            n_boot=3.1,
            n_sub=3.1,
            seed=101,
            col_sel=["mean"]
        )

        df_piped = df_stang >> gr.tf_bootstrap(
            tran=tran_stats,
            n_boot=3.1,
            n_sub=3.1,
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
