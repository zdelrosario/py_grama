import unittest

from context import grama as gr
from context import data
from context import models

## Test autoplots
##################################################
class TestAutoplot(unittest.TestCase):

    def setUp(self):
        self.md = models.make_test()
        self.md_rand = (
            gr.Model()
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(
                    z=df.x + df.y,
                ),
                var=["x", "y"],
                out=["z"],
            )
            >> gr.cp_marginals(
                x=gr.marg_mom("norm", mean=0, sd=1),
                y=gr.marg_mom("norm", mean=0, sd=1),
            )
            >> gr.cp_copula_independence()
        )

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
        )
        self.df_contour_aux = gr.eval_contour(
            self.md,
            var=["x0", "x1"],
            out=["y0"],
            df=gr.df_make(x2=[0, 1]),
        )

        self.df_sobol = (
            self.md_rand
            >> gr.ev_hybrid(df_det="nom")
            >> gr.tf_sobol()
        )

    def test_autoplot(self):
        # Full color
        gr.plot_auto(self.df_mc)
        gr.plot_auto(self.df_mc_skip)
        gr.plot_auto(self.df_sinew)
        gr.plot_auto(self.df_sinew_skip)
        gr.plot_auto(self.df_contour)
        gr.plot_auto(self.df_sobol)
        # Black & White
        gr.plot_auto(self.df_mc, color="bw")
        gr.plot_auto(self.df_mc_skip, color="bw")
        gr.plot_auto(self.df_sinew, color="bw")
        gr.plot_auto(self.df_sinew_skip, color="bw")
        gr.plot_auto(self.df_contour, color="bw")
        gr.plot_auto(self.df_sobol, color="bw")

        ## iocorr fits into pipeline
        gr.plot_auto(
            self.md_rand
            >> gr.ev_sample(n=10, df_det="nom")
            >> gr.tf_iocorr()
        )

        with self.assertRaises(ValueError):
            gr.plot_auto(self.df_contour_aux)
