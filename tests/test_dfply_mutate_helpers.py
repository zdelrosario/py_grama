import unittest

from context import grama as gr
from context import data

##==============================================================================
## mask helper tests
##==============================================================================


class TestFactors(unittest.TestCase):
    def setUp(self):
        pass

    def test_fct_reorder(self):
        ang_fct = gr.fct_reorder(data.df_stang.ang, data.df_stang.E)

        self.assertTrue(list(ang_fct.categories) == [0, 90, 45])
