import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr

## Test cohort shapley
##################################################
class TestCohortShapley(unittest.TestCase):
    def setUp(self):
        pass

    def test_cohort_shapley(self):
        df_data = gr.df_make(x0=[0, 0, 1, 1], x1=[0, 1, 0, 1], f=[0, 1, 1, 2],)

        df_true = gr.df_make(
            f_x0=[-0.5, -0.5, +0.5, +0.5], f_x1=[-0.5, +0.5, -0.5, +0.5],
        )

        df_cohort = gr.tran_shapley_cohort(df_data, var=["x0", "x1"], out=["f"])

        self.assertTrue(gr.df_equal(df_true, df_cohort))
