import unittest
import io
import sys

from context import grama as gr
from numpy import eye, zeros, sqrt
from numpy.random import multivariate_normal
from pandas import DataFrame

DF = gr.Intention()

## Test support points
##################################################
class TestReweight(unittest.TestCase):
    def setUp(self):
        self.md = (
            gr.Model()
            >> gr.cp_function(
                fun=lambda x: x,
                var=["x"],
                out=["y"],
            )
            >> gr.cp_marginals(x=dict(dist="norm", loc=0, scale=1))
            >> gr.cp_copula_independence()
        )

    def test_tran_reweight(self):
        """Test the functionality of tran_reweight()

        """
        ## Correctness
        # Choose scale based on Owen (2013) Exercise 9.7
        md_new = (
            self.md
            >> gr.cp_marginals(x=dict(dist="norm", loc=0, scale=sqrt(4/5)))
        )

        df_base = (
            self.md
            >> gr.ev_sample(n=500, df_det="nom", seed=101)
        )

        df = (
            df_base
            >> gr.tf_reweight(md_base=self.md, md_new=md_new)
            >> gr.tf_summarize(
                mu=gr.mean(DF.y * DF.weight),
                se=gr.sd(DF.y * DF.weight) / gr.sqrt(gr.n(DF.weight)),
                se_orig=gr.sd(DF.y) / gr.sqrt(gr.n(DF.weight)),
            )
        )
        mu = df.mu[0]
        se = df.se[0]
        se_orig = df.se_orig[0]

        self.assertTrue(
            mu - se * 2 < 0 and
            0 < mu + se * 2
        )

        ## Optimized IS should be more precise than ordinary monte carlo
        # print("se_orig = {0:4.3f}".format(se_orig))
        # print("se      = {0:4.3f}".format(se))
        self.assertTrue(se < se_orig)

        ## Invariants
        # Missing input in data
        with self.assertRaises(ValueError):
            gr.tran_reweight(df_base[["y"]], md_base=self.md, md_new=self.md)
        # Input mismatch
        with self.assertRaises(ValueError):
            gr.tran_reweight(
                df_base,
                md_base=self.md,
                md_new=gr.Model()
            )
        # Weights collision
        with self.assertRaises(ValueError):
            gr.tran_reweight(
                df_base
                >> gr.tf_mutate(weight=0),
                md_base=self.md,
                md_new=self.md
            )
