import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr
from context import models

## Test the built-in models
##################################################
class TestModels(unittest.TestCase):

    def setUp(self):
        pass

    def test_make(self):
        ## Models build
        md_cantilever_beam = models.make_cantilever_beam()
        md_ishigami = models.make_ishigami()
        md_linear_normal = models.make_linear_normal()
        md_piston = models.make_piston()
        md_piston_rand = models.make_piston_rand()
        md_plane_laminate = models.make_composite_plate_tension([0])
        md_plate_buckling = models.make_plate_buckle()
        md_poly = models.make_poly()
        md_prlc = models.make_prlc()
        md_prlc_rand = models.make_prlc_rand()
        md_test = models.make_test()
        md_trajectory_linear = models.make_trajectory_linear()

        ## Models evaluate
        df_cantilever = md_cantilever_beam >> gr.ev_nominal(df_det="nom")
        df_ishigami = md_ishigami >> gr.ev_nominal(df_det="nom")
        df_ln = md_linear_normal >> gr.ev_nominal(df_det="nom")
        df_piston = md_piston >> gr.ev_nominal(df_det="nom")
        df_piston_rand = md_piston_rand >> gr.ev_nominal(df_det="nom")
        df_plane = md_plane_laminate >> gr.ev_nominal(df_det="nom")
        df_plate = md_plate_buckling >> gr.ev_nominal(df_det="nom")
        df_poly = md_poly >> gr.ev_nominal(df_det="nom")
        df_prlc = md_prlc >> gr.ev_nominal(df_det="nom")
        df_prlc_rand = md_prlc_rand >> gr.ev_nominal(df_det="nom")
        df_test = md_test >> gr.ev_nominal(df_det="nom")
        df_traj = md_trajectory_linear >> gr.ev_nominal(df_det="nom")

        ## Piston models give approximately same output
        self.assertTrue(abs(df_piston.t_cyc[0] - df_piston_rand.t_cyc[0]) < 1e-6)

    def test_sir(self):
        from numpy import real
        from scipy.special import lambertw
        ## Verification test
        # Test parameters
        I0 = 1
        S0 = 99
        R0 = 0
        beta = 0.5
        gamma = 0.2

        # Asymptotic solution parameters
        N = I0 + S0 + R0
        R_0 = beta / gamma
        s_0 = S0 / N
        r_0 = R0 / N

        # Asymptotic solution
        S_inf = real(-(1/R_0) * lambertw(-s_0 * R_0 * np.exp(-R_0 * (1 - r_0))) * N)

        ## Base tolerance
        md_sir = models.make_sir()
        df_inf = gr.eval_df(
            md_sir,
            gr.df_make(
                t=1e6, # Approximation of t -> +\infty
                I0=I0,
                N=N,
                beta=beta,
                gamma=gamma,
            )
        )
        S_inf_comp = df_inf.S.values[-1]

        # Check relative tolerance
        self.assertTrue(abs(S_inf - S_inf_comp) / S_inf < 1e-3)
        self.assertTrue(abs(S_inf - S_inf_comp) / S_inf > 1e-5)

        ## Refined tolerance
        md_sir = models.make_sir(rtol=1e-6)
        df_inf = gr.eval_df(
            md_sir,
            gr.df_make(
                t=1e6, # Approximation of t -> +\infty
                I0=I0,
                N=N,
                beta=beta,
                gamma=gamma,
            )
        )
        S_inf_comp = df_inf.S.values[-1]

        # Check relative tolerance
        self.assertTrue(abs(S_inf - S_inf_comp) / S_inf < 1e-5)
