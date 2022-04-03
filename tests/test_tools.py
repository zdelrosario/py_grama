import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import models, data

## Core function tests
##################################################
class TestPipe(unittest.TestCase):
    def setUp(self):
        self.md = models.make_test()

    def test_pipe(self):
        ## Chain
        res = self.md >> gr.ev_hybrid(df_det="nom") >> gr.tf_sobol()
