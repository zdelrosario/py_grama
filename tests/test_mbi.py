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
        """Checks that a blank model is valid"""

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

        self.assertEqual(
            md_new0.var,
            ["x0"]
        )
        self.assertEqual(
            md_new0.out,
            ["y0"]
        )

        self.assertEqual(
            set(md_new1.var),
            set(["x0", "x1"])
        )
        self.assertEqual(
            set(md_new1.out),
            set(["y0", "y1"])
        )

        self.assertEqual(
            set(md_named.out),
            set(["bar1", "bar2"])
        )
        self.assertEqual(
            md_named.functions[0].name,
            "test"
        )
