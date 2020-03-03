import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import data

##==============================================================================
## grouping test functions
##==============================================================================
class TestGroup(unittest.TestCase):
    def test_group_attributes(self):
        d = data.df_diamonds >> gr.tf_group_by("cut")
        self.assertTrue(hasattr(d, "_grouped_by"))
        self.assertTrue(
            d._grouped_by == ["cut",]
        )
