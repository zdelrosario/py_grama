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

        self.assertTrue(isinstance(self.md.domain, gr.Domain))

        self.assertTrue(isinstance(self.md.density, gr.Density))

    def test_getvars(self):
        # Define test function
        def fun(x, y, z):
            return x + y + z
        # Check that getvars function works correctly
        self.assertTrue(gr.getvars(fun), ("x", "y", "z"))

    def test_freeze(self):
        md_base = (
            gr.Model()
            >> gr.cp_vec_function(
                fun = lambda df: gr.df_make(
                    f=df.x + df.y + df.z
                ),
                var=["x", "y", "z"],
                out=["f"],
            )
            >> gr.cp_bounds(z=(-1, +1))
            >> gr.cp_marginals(
                x=gr.marg_mom("norm", mean=0, sd=1),
                y=gr.marg_mom("norm", mean=0, sd=1),
            )
            >> gr.cp_copula_gaussian(
                df_corr=gr.df_make(var1="x", var2="y", corr=0.5)
            )
        )

        md_frz = (
            md_base
            >> gr.cp_freeze(x=0)
        )

        # Correct variable list
        self.assertTrue(set(md_frz.var) == {"y", "z"})

        # Correct evaluation
        df_res = gr.eval_df(md_frz, gr.df_make(y=[0, 1], z=0))
        self.assertTrue(all(df_res["x"] == 0))

        # Can still sample
        df_samp = gr.eval_sample(md_frz, df_det="nom", n=10)

        # Freezing non-existent variables raises error
        with self.assertRaises(ValueError):
            gr.comp_freeze(md_base, foo=0)

        # Freezing with multiple-values raises error
        with self.assertRaises(ValueError):
            gr.comp_freeze(md_base, x=[0, 1])

        # Dataframe input
        md_frz_df = (
            md_base
            >> gr.cp_freeze(df=gr.df_make(x=0, z=0))
        )
        self.assertTrue(set(md_frz_df.var) == {"y"})
        df_res = gr.eval_sample(md_frz_df, n=10, df_det="nom")

    def test_comp_function(self):
        """Test comp_function()"""

        md_new0 = gr.comp_function(self.md, fun=lambda x: x, var=1, out=1)
        md_new1 = gr.comp_function(md_new0, fun=lambda x: x, var=1, out=1)

        ## Operations above should not affect self.md
        md_named = gr.comp_function(
            self.md,
            fun=lambda x: [x, 2 * x],
            var=["foo"],
            out=["bar1", "bar2"],
            name="test",
        )

        ## Default var and out names
        self.assertEqual(md_new0.var, ["x0"])
        self.assertEqual(md_new0.out, ["y0"])

        ## New default names iterate counter
        self.assertEqual(set(md_new1.var), set(["x0", "x1"]))
        self.assertEqual(set(md_new1.out), set(["y0", "y1"]))

        ## Output names assigned correctly
        # Also tests for copy issues
        self.assertEqual(set(md_named.out), set(["bar1", "bar2"]))
        ## Function name assigned correctly
        self.assertEqual(md_named.functions[0].name, "test")

        ## Invariant checks
        with self.assertRaises(ValueError):
            # Missing function
            gr.comp_function(self.md, fun=None, var=["foo"], out=["bar"])
        with self.assertRaises(ValueError):
            # Missing var
            gr.comp_function(self.md, fun=lambda x: x, var=None, out=["bar"])
        with self.assertRaises(ValueError):
            # Missing out
            gr.comp_function(self.md, fun=lambda x: x, var=["foo"], out=None)
        with self.assertRaises(ValueError):
            # Intersection var / out names
            self.md >> gr.cp_function(lambda x: x, var=["x"], out=["x"], name="f0")
        with self.assertRaises(ValueError):
            # Non-unique function names
            self.md >> gr.cp_function(
                lambda x: x, var=1, out=1, name="f0"
            ) >> gr.cp_function(lambda x: x, var=1, out=1, name="f0")
        ## DAG invariant checks
        with self.assertRaises(ValueError):
            # Cycle by input
            self.md >> gr.cp_function(
                fun=lambda x0: x0, var=["y0"], out=1
            ) >> gr.cp_function(fun=lambda x0: x0, var=1, out=["y0"])
        with self.assertRaises(ValueError):
            # Non-unique output
            self.md >> gr.cp_function(
                fun=lambda x0: x0, var=1, out=["y0"]
            ) >> gr.cp_function(fun=lambda x0: x0, var=1, out=["y0"])

        ## Check vectorized builder
        md_vec = gr.comp_vec_function(
            self.md, fun=lambda df: df.assign(y0=df.x0), var=1, out=1,
        )
        self.assertTrue(
            gr.df_equal(gr.df_make(x0=0, y0=0), md_vec >> gr.ev_df(df=gr.df_make(x0=0)))
        )

    def test_comp_model(self):
        """Test model composition"""
        md_inner = (
            gr.Model("inner")
            >> gr.cp_function(fun=lambda x0, x1: x0 + x1, var=2, out=1)
            >> gr.cp_marginals(x0=dict(dist="norm", loc=0, scale=1))
            >> gr.cp_copula_independence()
        )

        ## Deterministic composition
        md_det = gr.Model("outer_det") >> gr.cp_md_det(md=md_inner)

        self.assertTrue(set(md_det.var) == {"x0", "x1"})
        self.assertTrue(md_det.out == ["y0"])
        gr.eval_df(md_det, df=gr.df_make(x0=0, x1=0))

        ## Deterministic composition
        md_sample = gr.Model("outer_det") >> gr.cp_md_sample(
            md=md_inner, param=dict(x0=("loc", "scale"))
        )

        self.assertTrue(set(md_sample.var) == {"x0_loc", "x0_scale", "x1"})
        self.assertTrue(set(md_sample.out) == {"y0"})
        gr.eval_df(md_sample, df=gr.df_make(x0_loc=0, x0_scale=1, x1=0))

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
        md1 = gr.comp_marginals(self.md, x={"dist": "norm", "loc": 0, "scale": 1})

        self.assertEqual(md1.density.marginals["x"].d_name, "norm")

        ## Check sign and default sign
        md2 = gr.comp_marginals(
            self.md,
            x={"dist": "norm", "loc": 0, "scale": 1},
            y={"dist": "norm", "loc": 0, "scale": 1, "sign": -1},
        )

        self.assertEqual(md2.density.marginals["x"].sign, 0)
        self.assertEqual(md2.density.marginals["y"].sign, -1)

        ## Check number and list of vars computed correctly
        md3 = gr.comp_bounds(self.md, x=(0, 1), y=(0, 1))
        md3 = gr.comp_marginals(md3, x={"dist": "uniform", "loc": 0, "scale": 1})

        self.assertEqual(md3.n_var, 2)
        self.assertEqual(md3.n_var_det, 1)
        self.assertEqual(md3.n_var_rand, 1)
        self.assertEqual(md3.var_det, ["y"])
        self.assertEqual(md3.var_rand, ["x"])

        ## Invariant raises
        with self.assertRaises(NotImplementedError):
            gr.comp_marginals(self.md, x={})

    def test_comp_copula(self):
        md_incomplete = gr.comp_marginals(
            self.md,
            x={"dist": "uniform", "loc": -1, "scale": 2},
            y={"dist": "uniform", "loc": -1, "scale": 2},
        )

        with self.assertRaises(AttributeError):
            md_incomplete.sample()

        ## Independence copula
        md_independence = gr.comp_copula_independence(md_incomplete)
        df_ind = md_independence.density.sample()
        self.assertTrue(set(df_ind.columns) == set(["x", "y"]))

        ## Gaussian copula
        md_gaussian = gr.comp_copula_gaussian(
            md_incomplete, pd.DataFrame(dict(var1=["x"], var2=["y"], corr=[0.5]))
        )
        df_gau = md_gaussian.density.sample()
        self.assertTrue(set(df_gau.columns) == set(["x", "y"]))
