import unittest
import pandas as pd

from context import grama as gr
DF = gr.Intention()

class TestDataHelpers(unittest.TestCase):
    def setUp(self):
        pass

    def test_df_equal(self):
        df1 = pd.DataFrame(dict(x=[0], y=[0]))
        df2 = pd.DataFrame(dict(x=[0]))

        self.assertTrue(gr.df_equal(df1, df1))
        self.assertTrue(gr.df_equal(df1, df2) == False)

    def test_df_make(self):
        # Check correctness
        df_true = pd.DataFrame(dict(x=[0, 1], y=[0, 0], z=[1, 1]))
        df_res = gr.df_make(x=[0, 1], y=[0], z=1)

        self.assertTrue(gr.df_equal(df_true, df_res))

        # Check for mismatch
        with self.assertRaises(ValueError):
            gr.df_make(x=[1, 2, 3], y=[1, 2])

        # Catch an intention operator
        with self.assertRaises(ValueError):
            gr.df_make(y=DF.x)

    def test_df_grid(self):
        # Check correctness
        df_res = gr.df_grid(
            x=[0, 1, 2],
            y=["A", "B"],
            z=1,
            w="x",
        )
        df_true = pd.DataFrame(dict(
            x=[  0,   1,   2,   0,   1,   2],
            y=["A", "A", "A", "B", "B", "B"],
            z=[  1,   1,   1,   1,   1,   1],
            w=["x", "x", "x", "x", "x", "x"],
        ))

        self.assertTrue(gr.df_equal(df_true, df_res))

        # Catch an intention operator
        with self.assertRaises(ValueError):
            gr.df_grid(y=DF.x)
