import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr

## Test support points
##################################################
class TestSupportPoints(unittest.TestCase):
    def setUp(self):
        n = 100
        self.df = pd.DataFrame(
            data=np.random.multivariate_normal(np.zeros(3), np.eye(3), size=n,),
            columns=["x", "y", "z"],
        )
        self.df["c"] = list(map(lambda x: str(x), range(n)))

    def test_tran_sp(self):
        """Test the functionality of tran_sp()

        Note that *correctness* is verified elsewhere;
        see ./longrun/sp_convergence.ipynb
        """

        ## Basic facts
        df_sp = gr.tran_sp(self.df, n=10,)
        # Correct number of samples
        self.assertTrue(df_sp.shape[0] == 10)
        # Correct variables (numeric only)
        self.assertTrue(set(df_sp.columns) == {"x", "y", "z"})

        ## Subsetting target variables
        df_xy = gr.tran_sp(self.df, n=10, var=["x", "y"],)
        # Subset correctly
        self.assertTrue(set(df_xy.columns) == {"x", "y"})
