import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr
from context import models

## Test the Model Building Interface
##################################################
class TestMBI(unittest.TestCase):

    def setUp(self):
        self.md = gr.Model()

    def test_blank_model(self):
        """Checks that blank model is valid"""

        # Capture printpretty()
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.md.printpretty()
        sys.stdout = sys.__stdout__

        self.assertTrue(
            isinstance(self.md.domain, gr.Domain)
        )

        self.assertTrue(
            isinstance(self.md.density, gr.Density)
        )

    def test_comp_function(self):
        """Test comp_function()"""

        md_new0 = gr.comp_function(
            self.md,
            fun=lambda x: x,
            var=1,
            out=1
        )
        md_new1 = gr.comp_function(
            md_new0,
            fun=lambda x: x,
            var=1,
            out=1
        )

        ## Operations above should not affect self.md
        md_named = gr.comp_function(
            self.md,
            fun=lambda x: [x, 2*x],
            var=["foo"],
            out=["bar1", "bar2"],
            name="test"
        )

        ## Default var and out names
        self.assertEqual(
            md_new0.var,
            ["x0"]
        )
        self.assertEqual(
            md_new0.out,
            ["y0"]
        )

        ## New default names iterate counter
        self.assertEqual(
            set(md_new1.var),
            set(["x0", "x1"])
        )
        self.assertEqual(
            set(md_new1.out),
            set(["y0", "y1"])
        )

        ## Output names assigned correctly
        # Also tests for copy issues
        self.assertEqual(
            set(md_named.out),
            set(["bar1", "bar2"])
        )
        ## Function name assigned correctly
        self.assertEqual(
            md_named.functions[0].name,
            "test"
        )

    def test_comp_bounds(self):
        """Test comp_bounds()"""

        ## Add bound w/o function
        md1 = gr.comp_bounds(self.md, x=(0, 1))

        self.assertEqual(md1.domain.bounds["x"], [0, 1])

        ## Add bound with function
        md2 = gr.comp_function(self.md, fun=lambda x: x, var=["x"], out=["y"])
        md2 = gr.comp_bounds(md2, x=(0, 1))

        self.assertEqual(md2.n_var, 1)

        ## Add multiple bounds
        md3 = gr.comp_function(self.md, fun=lambda x: x, var=2, out=2)
        md3 = gr.comp_bounds(md3, x0=(0, 1), x1=(0, 1))

        self.assertEqual(md3.n_var, 2)

        ## Additional bound values ignored
        md4 = gr.comp_bounds(self.md, x=(0, 1, 2))

        self.assertEqual(md4.domain.bounds["x"][0], 0)
        self.assertEqual(md4.domain.bounds["x"][1], 1)

    def test_comp_marginals(self):
        """Test comp_bounds()"""

        ## Add marginal w/o function
        md1 = gr.comp_marginals(
            self.md,
            x={"dist": "norm", "loc": 0, "scale": 1}
        )

        self.assertEqual(
            md1.density.marginals["x"].d_name,
            "norm"
        )

        ## Check sign and default sign
        md2 = gr.comp_marginals(
            self.md,
            x={"dist": "norm", "loc": 0, "scale": 1},
            y={"dist": "norm", "loc": 0, "scale": 1, "sign": -1}
        )

        self.assertEqual(
            md2.density.marginals["x"].sign,
            0
        )
        self.assertEqual(
            md2.density.marginals["y"].sign,
            -1
        )

        ## Check number and list of vars computed correctly
        md3 = gr.comp_bounds(self.md, x=(0, 1), y=(0, 1))
        md3 = gr.comp_marginals(md3, x={"dist": "uniform", "loc": 0, "scale": 1})

        self.assertEqual(md3.n_var, 2)
        self.assertEqual(md3.n_var_det, 1)
        self.assertEqual(md3.n_var_rand, 1)
        self.assertEqual(md3.var_det, ["y"])
        self.assertEqual(md3.var_rand, ["x"])
