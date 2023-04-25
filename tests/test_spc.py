import unittest

from context import grama as gr
from context import data
DF = gr.Intention()

## Core function tests
##################################################
class TestSPC(unittest.TestCase):
    """Test statistical process control (SPC) tools
    """

    def test_factors(self):
        """Test factor implementations"""
        ## Standard deviation de-biasing constant
        self.assertAlmostEqual(
            gr.c_sd(2),
            0.7979,
            places=4,
        )
        self.assertAlmostEqual(
            gr.c_sd(4),
            0.9213,
            places=4,
        )
        self.assertAlmostEqual(
            gr.c_sd(10),
            0.9727,
            places=4,
        )
        ## Standard deviation control limits
        self.assertAlmostEqual(
            gr.B3(5),
            0.0000,
            places=3,
        )
        self.assertAlmostEqual(
            gr.B3(6),
            0.030,
            places=3,
        )
        self.assertAlmostEqual(
            gr.B3(10),
            0.284,
            places=3,
        )

        self.assertAlmostEqual(
            gr.B4(5),
            2.089,
            places=3,
        )
        self.assertAlmostEqual(
            gr.B4(6),
            1.970,
            places=3,
        )
        self.assertAlmostEqual(
            gr.B4(10),
            1.716,
            places=3,
        )

    def test_plot_xbs(self):
        r"""Tests that Xbar and S chart runs"""
        ## Basic functionality
        (
            data.df_shewhart
            >> gr.tf_mutate(idx=DF.index // 10)
            >> gr.pt_xbs(group="idx", var="tensile_strength")
        )

        ## Works with discrete group variable
        (
            data.df_shewhart
            >> gr.tf_mutate(idx=gr.as_factor(DF.index // 10))
            >> gr.pt_xbs(group="idx", var="tensile_strength")
        )

        ## Black & White color functionality
        (
            data.df_shewhart
            >> gr.tf_mutate(idx=DF.index // 10)
            >> gr.pt_xbs(group="idx", var="tensile_strength", color="bw")
        )
