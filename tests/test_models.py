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
        md_plane_laminate = models.make_composite_plate_tension([0])
        md_plate_buckling = models.make_plate_buckle()
        md_poly = models.make_poly()
        md_test = models.make_test()

        ## Models evaluate
        md_cantilever_beam >> gr.ev_nominal(df_det="nom")
        md_ishigami >> gr.ev_nominal(df_det="nom")
        md_linear_normal >> gr.ev_nominal(df_det="nom")
        md_plane_laminate >> gr.ev_nominal(df_det="nom")
        md_plate_buckling >> gr.ev_nominal(df_det="nom")
        md_poly >> gr.ev_nominal(df_det="nom")
        md_test >> gr.ev_nominal(df_det="nom")
