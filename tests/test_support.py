import unittest
import io
import sys

from context import grama as gr
from numpy import eye, zeros
from numpy.random import multivariate_normal
from pandas import DataFrame

## Test support points
##################################################
class TestSupportPoints(unittest.TestCase):
    def setUp(self):
        n = 100
        self.df = DataFrame(
            data=multivariate_normal(zeros(3), eye(3), size=n), columns=["x", "y", "z"],
        )
        self.df["c"] = list(map(str, range(n)))

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
        df_xy = gr.tran_sp(self.df, n=10, var=["x", "y"])
        # Subset correctly
        self.assertTrue(set(df_xy.columns) == {"x", "y"})

        ## Warning raised on lack of convergence
        with self.assertWarns(RuntimeWarning):
            gr.tran_sp(self.df, n=10, var=["x", "y"], n_maxiter=1)
