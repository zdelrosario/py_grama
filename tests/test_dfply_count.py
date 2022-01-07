import numpy as np
import pandas as pd
import unittest

from context import grama as gr
from context import data

DF = gr.Intention()

##==============================================================================
## summarization test functions
##==============================================================================

class TestSummarize(unittest.TestCase):
    def test_summarize(self):
        df = gr.df_make(
            x=["A", "A", "B", "B"],
            y=["A", "B", "A", "B"],
        )
        df_true1 = gr.df_make(
            x=["A", "B"],
            n=[ 2,    2],
        )
        df_true2 = gr.df_make(
            x=["A", "A", "B", "B"],
            y=["A", "B", "A", "B"],
            n=[  1,   1,   1,   1],
        )

        df_res1 = (
            df
            >> gr.tf_count(DF.x)

        )
        self.assertTrue(df_true1.equals(df_res1))

        df_res2 = (
            df
            >> gr.tf_count(DF.x, DF.y)
        )
        self.assertTrue(df_true2.equals(df_res2))
