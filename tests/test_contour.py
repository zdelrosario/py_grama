import numpy as np
import pandas as pd
import unittest

from context import grama as gr

##################################################
class TestContour(unittest.TestCase):
    def setUp(self):
        pass

    def test_contour(self):
        md1 = (
            gr.Model()
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(
                    f=df.x**2 + df.y**2,
                    g=df.x + df.y,
                ),
                var=["x", "y"],
                out=["f", "g"],
            )
            >> gr.cp_bounds(
                x=(-1, +1),
                y=(-1, +1),
            )
        )

        md2 = (
            gr.Model()
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(
                    f=(df.c) * df.x + (1 - df.c) * df.y,
                ),
                var=["x", "y", "c"],
                out=["f"],
            )
            >> gr.cp_bounds(
                x=(-1, +1),
                y=(-1, +1),
                c=(+0, +1),
            )
        )

        ## Basic functionality
        df_res1 = (
            md1
            >> gr.ev_contour(
                var=["x", "y"],
                out=["f", "g"],
                n_side=10, # Coarse, for speed
            )
        )
        # Contains correct columns
        self.assertTrue("x" in df_res1.columns)
        self.assertTrue("y" in df_res1.columns)

        df_res2 = (
            md2
            >> gr.ev_contour(
                var=["x", "y"],
                out=["f"],
                df=gr.df_make(c=[0, 1]),
                n_side=10, # Coarse, for speed
            )
        )
        # Contains auxiliary variable
        self.assertTrue("c" in df_res2.columns)

        df_res3 = (
            md1
            >> gr.ev_contour(
                var=["x", "y"],
                out=["g"],
                levels=dict(g=[-1, 0, +1]),
                n_side=10, # Coarse, for speed
            )
        )
        # Correct manual levels
        self.assertTrue(set(df_res3.level) == {-1, 0, +1})

        # Drops redundant (swept) inputs under-the-hood
        (
            md2
            >> gr.ev_contour(
                var=["x", "y"],
                out=["f"],
                df=gr.eval_nominal(md2, df_det="nom"),
                n_side=10, # Coarse, for speed
            )
        )

        # Correct manual levels
        with self.assertWarns(Warning):
            df_res4 = (
                md1
                >> gr.ev_contour(
                    var=["x", "y"],
                    out=["g"],
                    levels=dict(g=[-1, 0, +1, +1e3]),
                    n_side=10, # Coarse, for speed
                )
            )
            # Correct manual levels
            self.assertTrue(set(df_res4.level) == {-1, 0, +1})

        ## Check assertions
        # No var
        with self.assertRaises(ValueError):
            res = (
                md1
                >> gr.ev_contour(out=["f"])
            )

        # Incorrect number of inputs
        with self.assertRaises(ValueError):
            res = (
                md1
                >> gr.ev_contour(var=["x", "y", "z"])
            )

        # Unavailable inputs
        with self.assertRaises(ValueError):
            res = (
                md1
                >> gr.ev_contour(var=["foo", "bar"])
            )

        # Unsupported input
        with self.assertRaises(ValueError):
            res = (
                md2
                >> gr.ev_contour(
                    var=["x", "y"],
                    out=["f"],
                )
            )

        with self.assertRaises(ValueError):
            res = (
                md2
                >> gr.ev_contour(
                    var=["x", "y"],
                    out=["f"],
                    df=gr.df_make(foo=1)
                )
            )

        # Zero bound width
        with self.assertRaises(ValueError):
           res = (
               gr.Model()
               >> gr.cp_vec_function(
                   fun=lambda df: gr.df_make(
                       f=df.x**2 + df.y**2,
                       g=df.x + df.y,
                   ),
                   var=["x", "y"],
                   out=["f", "g"],
               )
               >> gr.cp_bounds(
                   x=( 0,  0),
                   y=(-1, +1),
               )

               >> gr.ev_contour(
                   var=["x", "y"],
                   out=["f", "g"],
                   n_side=10, # Coarse, for speed
               )
           )

        # No out
        with self.assertRaises(ValueError):
            res = (
                md1
                >> gr.ev_contour(var=["x", "y"])
            )

        # Output unavailable
        with self.assertRaises(ValueError):
            res = (
                md1
                >> gr.ev_contour(var=["x", "y"], out=["foo"])
            )
