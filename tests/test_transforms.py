import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr
from context import data
from context import models

DF = gr.Intention()

## Test transform tools
##################################################
class TestTools(unittest.TestCase):
    def setUp(self):
        pass

    def test_tran_md(self):
        md = models.make_test()

        ## Check for identical responses
        df = gr.df_make(x0=1, x1=1, x2=1)
        df_ev = gr.eval_df(md, df=df)
        df_tf = gr.tran_md(df, md=md)

        self.assertTrue(gr.df_equal(df_ev, df_tf))

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

        # Empty cases
        gr.df_equal(
            df,
            gr.tran_outer(pd.DataFrame(), df_outer=df)
        )
        gr.df_equal(
            df,
            gr.tran_outer(df, df_outer=pd.DataFrame())
        )

    def test_gauss_copula(self):
        md = gr.Model() >> gr.cp_marginals(
            E=gr.marg_fit("norm", data.df_stang.E),
            mu=gr.marg_fit("beta", data.df_stang.mu),
            thick=gr.marg_fit("uniform", data.df_stang.thick),
        )
        df_corr = gr.tran_copula_corr(data.df_stang, model=md)

    def test_kfold(self):
        df_train = pd.DataFrame(
            dict(X=list(range(4)), Y=[0, 0, 1, 1], _fold=["a", "a", "b", "b"])
        )
        df_true = pd.DataFrame(
            dict(
                mse_Y=[
                    np.mean((np.array([0, 0]) - 1) ** 2),
                    np.mean((np.array([1, 1]) - 0) ** 2),
                ],
                _kfold=[0, 1],
            )
        )
        df_true_m = pd.DataFrame(
            dict(
                mse_Y=[
                    np.mean((np.array([0, 0]) - 1) ** 2),
                    np.mean((np.array([1, 1]) - 0) ** 2),
                ],
                _fold=["a", "b"],
            )
        )

        ## Unshuffled, auto-generated folds
        df_res = df_train >> gr.tf_kfolds(
            k=2,
            ft=gr.ft_rf(out=["Y"], var=["X"]),
            shuffle=False,
            summaries=dict(mse=gr.mse),
        )

        pd.testing.assert_frame_equal(
            df_res,
            df_true,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )

        ## Manual folds
        df_manual = df_train >> gr.tf_kfolds(
            ft=gr.ft_rf(out=["Y"], var=["X"]),
            var_fold="_fold",
            summaries=dict(mse=gr.mse),
        )

        pd.testing.assert_frame_equal(
            df_manual,
            df_true_m,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )


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

    def test_pca(self):
        df_test = pd.DataFrame(dict(x0=[1, 2, 3], x1=[1, 2, 3]))
        df_offset = pd.DataFrame(dict(x0=[1, 2, 3], x1=[3, 4, 5]))
        df_scaled = pd.DataFrame(dict(x0=[1, 2, 3], x1=[1, 3, 5]))

        df_true = pd.DataFrame(
            dict(
                lam=[2, 0],
                x0=[1 / np.sqrt(2), 1 / np.sqrt(2)],
                x1=[1 / np.sqrt(2), -1 / np.sqrt(2)],
            )
        )

        ## Check correctness
        df_pca = df_test >> gr.tf_pca()

        self.assertTrue(np.allclose(df_true.lam, df_pca.lam))

        ## Offset data should not affect results
        df_pca_off = df_offset >> gr.tf_pca()
        self.assertTrue(gr.df_equal(df_pca, df_pca_off))

        ## Check standardized form
        df_pca_scaled = df_test >> gr.tf_pca(standardize=True)
        self.assertTrue(gr.df_equal(df_pca, df_pca_scaled))

    def test_iocorr(self):
        df = (
            gr.df_make(x=[1., 2., 3., 4.])
            >> gr.tf_mutate(
                y=+0.5 * DF.x,
                z=-0.5 * DF.x,
            )
            >> gr.tf_iocorr(var=["x"], out=["y", "z"])
        )
        df_true = gr.df_make(
            var=["x", "x"],
            out=["y", "z"],
            rho=[1.0, -1.0],
        )

        ## Check for correct values
        self.assertTrue(gr.df_equal(df, df_true))


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


# --------------------------------------------------
class TestDR(unittest.TestCase):
    def setUp(self):
        pass

    def test_tsne(self):
        ## t-SNE executes successfully
        df_tsne = (
            data.df_diamonds >> gr.tf_sample(n=100) >> gr.tf_tsne(var=["x", "y", "z"])
        )

    def test_umap(self):
        ## UMAP executes successfully
        df_umap = (
            data.df_diamonds >> gr.tf_sample(n=100) >> gr.tf_umap(var=["x", "y", "z"])
        )


# --------------------------------------------------
# class TestFeaturize(unittest.TestCase):
#     def setUp(self):
#         pass

#     def test_tran_poly(self):
#         df = gr.df_make(x=[0.0, 1.0, 0.0], y=[0.0, 0.0, 1.0], z=[1.0, 2.0, 3.0],)
#         df_true = df.copy()
#         df_true["1"] = [1.0, 1.0, 1.0]
#         df_true["x^2"] = [0.0, 1.0, 0.0]
#         df_true["x y"] = [0.0, 0.0, 0.0]
#         df_true["y^2"] = [0.0, 0.0, 1.0]

#         df_res = gr.tran_poly(df, var=["x", "y"], degree=2, keep=True)
#         self.assertTrue(gr.df_equal(df_true, df_res[df_true.columns]))


# --------------------------------------------------
# class TestMatminer(unittest.TestCase):
#     def test_magpie_featurizer(self):
#         df_magpie_check = pd.DataFrame(
#             {
#                 "MagpieData minimum Number": [1.0],
#                 "MagpieData maximum Number": [8.0],
#                 "MagpieData range Number": [7.0],
#             }
#         )
#         df_test = gr.df_make(FORMULA=["C6H12O6"])

#         df_res = df_test >> gr.tf_feat_composition()

#         self.assertTrue(gr.df_equal(df_test, df_res[df_test.columns]))
