import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr
from context import data

## Test the built-in datasets
##################################################
class TestData(unittest.TestCase):

    def setUp(self):
        pass

    def test_make(self):
        df_stang = data.df_stang

    def test_install(self):
        # Only works if grama installed locally!
        from grama.data import df_stang
