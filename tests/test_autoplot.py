import unittest

from context import grama as gr
from context import data
from context import models

## Test autoplots
##################################################
class TestAutoplot(unittest.TestCase):

    def setUp(self):
        self.md = models.make_test()
        self.df_mc = gr.eval_sample(self.md, n=10, df_det="nom")
        self.df_mc_skip = gr.eval_sample(self.md, n=10, df_det="nom", skip=True)
        self.df_sinew = gr.eval_sinews(self.md, n_density=2, n_sweeps=1, df_det="nom")
        self.df_sinew_skip = gr.eval_sinews(
            self.md,
            n_density=2,
            n_sweeps=1,
            df_det="nom",
            skip=True
        )

        self.df_contour = gr.eval_contour(
            self.md,
            var=["x0", "x1"],
            out=["y0"],
            df=gr.df_make(x2=0),
            n_side=32,
        )

    def test_autoplot(self):
        gr.plot_auto(self.df_mc)
        gr.plot_auto(self.df_mc_skip)
        gr.plot_auto(self.df_sinew)
        gr.plot_auto(self.df_sinew_skip)
        gr.plot_auto(self.df_contour)
