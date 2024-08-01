import unittest
from os.path import join
from context import grama as gr
from context import models

## Core function tests
##################################################
class TestPipe(unittest.TestCase):
    def setUp(self):
        self.md = models.make_test()

    def test_pipe(self):
        ## Chain
        res = self.md >> gr.ev_hybrid(df_det="nom") >> gr.tf_sobol()

class TestFindFiles(unittest.TestCase):
    def test_find_files(self):
        # Non-recursive
        res = gr.find_files(".", ".py", recursive=False)
        self.assertTrue(
            join(".", "context.py") in res
        )
        self.assertTrue(
            not join("longrun", "sp_convergence.ipynb") in res
        )
        # Handles missing ext dot
        res = gr.find_files(".", "py", recursive=False)
        self.assertTrue(
            join(".", "context.py") in res
        )
        # No return
        res = gr.find_files(".", ".exe", recursive=False)
        self.assertTrue(len(res) == 0)
        # Recursive
        res = gr.find_files(".", ".ipynb", recursive=True)
        self.assertTrue(
            join(".", "longrun", "sp_convergence.ipynb") in res
        )
        # Accepts paths to join
        res = gr.find_files([".", "longrun"], ".ipynb", recursive=False)
        self.assertTrue(
            join(".", "longrun", "sp_convergence.ipynb") in res
        )
