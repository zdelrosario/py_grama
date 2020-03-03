import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr
from context import data
from context import models

## Test transform tools
##################################################
class TestTools(unittest.TestCase):
    def setUp(self):
        pass

    def test_bootstrap(self):
        df_stang = data.df_stang
        df_stang._meta = "foo"

        def tran_stats(df):
            val = df.select_dtypes(include="number").values

            means = np.mean(val, axis=0)
            stds = np.std(val, axis=0)

            # Check metadata propagation
            self.assertTrue(df._meta == "foo")

            return pd.DataFrame(
                data={
                    "var": df.select_dtypes(include="number").columns,
                    "mean": means,
                    "std": stds,
                }
            )

        df_res = gr.tran_bootstrap(
            df_stang, tran=tran_stats, n_boot=3e0, n_sub=3e0, seed=101
        )

        df_sel = gr.tran_bootstrap(
            df_stang, tran=tran_stats, n_boot=3e0, n_sub=3e0, seed=101, col_sel=["mean"]
        )

        df_piped = df_stang >> gr.tf_bootstrap(
            tran=tran_stats, n_boot=3e0, n_sub=3e0, seed=101
        )

        ## Test output shape
        self.assertTrue(
            set(df_res.columns)
            == set(["var", "mean", "mean_lo", "mean_up", "std", "std_lo", "std_up"])
        )
        self.assertTrue(df_res.shape[0] == 4)

        self.assertTrue(
            set(df_sel.columns) == set(["var", "mean", "mean_lo", "mean_up", "std"])
        )
        self.assertTrue(df_sel.shape[0] == 4)

        ## Test pipe
        self.assertTrue(gr.df_equal(df_res, df_piped))

    def test_outer(self):
        df = pd.DataFrame(dict(x=[1, 2]))
        df_outer = pd.DataFrame(dict(y=[3, 4]))

        df_true = pd.DataFrame(dict(x=[1, 2, 1, 2], y=[3, 3, 4, 4]))

        df_res = gr.tran_outer(df, df_outer)
        df_piped = df >> gr.tf_outer(df_outer)

        pd.testing.assert_frame_equal(
            df_true,
            df_res,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )
        pd.testing.assert_frame_equal(
            df_piped,
            df_res,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )

    def test_gauss_copula(self):
        md = gr.Model() >> gr.cp_marginals(
            E=gr.continuous_fit(data.df_stang.E, "norm"),
            mu=gr.continuous_fit(data.df_stang.mu, "beta"),
            thick=gr.continuous_fit(data.df_stang.thick, "uniform"),
        )
        df_corr = gr.tran_copula_corr(data.df_stang, model=md)


# --------------------------------------------------
class TestSummaries(unittest.TestCase):
    def setUp(self):
        self.md = models.make_test()

    def test_sobol(self):
        ## Minimal
        df_first = gr.eval_hybrid(self.md, df_det="nom")
        df_sobol = gr.tran_sobol(df_first)
        self.assertTrue(set(df_sobol.columns) == set(["y0", "ind"]))
        self.assertTrue(set(df_sobol["ind"]) == set(["S_x0", "S_x1"]))

        ## Full
        df_full = gr.tran_sobol(df_first, full=True)
        self.assertTrue(set(df_full.columns) == set(["y0", "ind"]))
        self.assertTrue(
            set(df_full["ind"]) == set(["S_x0", "S_x1", "T_x0", "T_x1", "var"])
        )

        ## Total order
        df_total = gr.eval_hybrid(self.md, df_det="nom", plan="total")
        df_sobol_total = gr.tran_sobol(df_total)
        self.assertTrue(set(df_sobol.columns) == set(["y0", "ind"]))
        self.assertTrue(set(df_sobol["ind"]) == set(["S_x0", "S_x1"]))


# --------------------------------------------------
class TestAsub(unittest.TestCase):
    def setUp(self):
        pass

    def test_asub(self):
        df_data = pd.DataFrame(
            dict(Df_Dx=[1 / np.sqrt(2)] * 2, Df_Dy=[1 / np.sqrt(2)] * 2)
        )
        df_true = pd.DataFrame(
            dict(
                x=[+1 / np.sqrt(2), +1 / np.sqrt(2)],
                y=[+1 / np.sqrt(2), -1 / np.sqrt(2)],
                out=["f", "f"],
                lam=[1, 0],
            )
        )

        df_res = gr.tran_asub(df_data)

        ## Entries correct
        angles = gr.tran_angles(df_true[["x", "y"]], df_res[["x", "y"]])
        self.assertTrue(np.allclose(angles, [0, 0]))

        ## Expected columns
        self.assertTrue(set(df_res.columns) == set(df_true.columns))

        df_piped = df_data >> gr.tf_asub()
        self.assertTrue(df_res.equals(df_piped))


# --------------------------------------------------
class TestAngles(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(dict(v=[1, 1]))
        self.df_v1 = pd.DataFrame(dict(w=[+1, -1]))
        self.df_v2 = pd.DataFrame(dict(w=[+1, +1]))

    def test_angles(self):
        theta1 = gr.tran_angles(self.df, self.df_v1)
        theta2 = gr.tran_angles(self.df, self.df_v2)

        self.assertTrue(np.isclose(theta1, np.pi / 2))
        self.assertTrue(np.isclose(theta2, 0))

        theta_piped = self.df >> gr.tf_angles(self.df_v1)
        self.assertTrue(np.isclose(theta1, theta_piped))

        with self.assertRaises(ValueError):
            gr.tran_angles(pd.DataFrame(), self.df_v1)

        with self.assertRaises(ValueError):
            gr.tran_angles(self.df, pd.DataFrame())


# --------------------------------------------------
class TestInner(unittest.TestCase):
    def setUp(self):
        pass

    def test_inner(self):
        df = pd.DataFrame(dict(v=[+1, +1], w=[-1, +1]))

        df_w = pd.DataFrame(dict(v=[1, 0, 1], w=[0, 1, 1], id=["v", "w", "x"]))
        df_true = df.copy()
        df_true["dot_v"] = df.v
        df_true["dot_w"] = df.w
        df_true["dot_x"] = df.v + df.w

        ## Core functionality
        df_noappend = gr.tran_inner(df, df_w, name="id", append=False)

        pd.testing.assert_frame_equal(
            df_true[["dot_v", "dot_w", "dot_x"]],
            df_noappend,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )

        df_res = gr.tran_inner(df, df_w, name="id")
        pd.testing.assert_frame_equal(
            df_true,
            df_res,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )

        ## Pipe
        df_piped = df >> gr.tf_inner(df_w, name="id")
        self.assertTrue(df_res.equals(df_piped))

        ## Small weights
        df_w_small = pd.DataFrame(dict(v=[1], w=[1]))
        df_true_small = df.copy()
        df_true_small["dot"] = df.v + df.w

        df_res_small = gr.tran_inner(df, df_w_small)
        pd.testing.assert_frame_equal(
            df_true_small,
            df_res_small,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )

        ## No name column
        df_w_noname = df_w.drop("id", axis=1)
        df_true_noname = df.copy()
        df_true_noname["dot0"] = df.v
        df_true_noname["dot1"] = df.w
        df_true_noname["dot2"] = df.v + df.w

        df_res_noname = gr.tran_inner(df, df_w_noname)
        pd.testing.assert_frame_equal(
            df_true_noname,
            df_res_noname,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )
