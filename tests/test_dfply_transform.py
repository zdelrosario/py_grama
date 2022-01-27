import numpy as np
import pandas as pd
import unittest

from context import grama as gr
from context import data

X = gr.Intention()

##==============================================================================
## transform test functions
##==============================================================================
def group_mutate_helper(df):
    df["testcol"] = df["x"] * df.shape[0]
    return df


class testTransform(unittest.TestCase):
    def test_mutate(self):
        df = data.df_diamonds.copy()
        df["testcol"] = 1
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_mutate(testcol=1)))
        df["testcol"] = df["x"]
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_mutate(testcol=X.x)))
        df["testcol"] = df["x"] * df["y"]
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_mutate(testcol=X.x * X.y)))
        df["testcol"] = df["x"].mean()
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_mutate(testcol=np.mean(X.x)))
        )

    def test_group_mutate(self):
        df = data.df_diamonds.copy()
        df = df.groupby("cut").apply(group_mutate_helper)
        d = (
            data.df_diamonds
            >> gr.tf_group_by("cut")
            >> gr.tf_mutate(testcol=X.x * X.shape[0])
            >> gr.tf_ungroup()
        )
        self.assertTrue(df.equals(d.sort_index()))

    def test_mutate_if(self):
        df = data.df_diamonds.copy()
        for col in df:
            try:
                if max(df[col]) < 10:
                    df[col] *= 2
            except:
                pass
        self.assertTrue(
            df.equals(
                data.df_diamonds
                >> gr.tf_mutate_if(lambda col: max(col) < 10, lambda row: row * 2)
            )
        )
        df = data.df_diamonds.copy()
        for col in df:
            try:
                if any(df[col].str.contains(".")):
                    df[col] = df[col].str.lower()
            except:
                pass
        self.assertTrue(
            df.equals(
                data.df_diamonds
                >> gr.tf_mutate_if(
                    lambda col: any(col.str.contains(".")), lambda row: row.str.lower()
                )
            )
        )
        df = data.df_diamonds.copy()
        for col in df:
            try:
                if min(df[col]) < 1 and mean(df[col]) < 4:
                    df[col] *= -1
            except:
                pass
        self.assertTrue(
            df.equals(
                data.df_diamonds
                >> gr.tf_mutate_if(
                    lambda col: min(col) < 1 and mean(col) < 4, lambda row: -row
                )
            )
        )
