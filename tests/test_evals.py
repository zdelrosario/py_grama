import numpy as np
import pandas as pd
import unittest

from collections import OrderedDict as od
from context import core
from context import grama as gr
from context import models
from context import ev
from pyDOE import lhs

##################################################
class TestDefaults(unittest.TestCase):
    def setUp(self):
        # 2D identity model with permuted df inputs
        domain_2d = gr.Domain(bounds={"x": [-1.0, +1], "y": [0.0, 1.0]})
        marginals = {}
        marginals["x"] = gr.MarginalNamed(
            d_name="uniform", d_param={"loc": -1, "scale": 2}
        )
        marginals["y"] = gr.MarginalNamed(
            sign=-1, d_name="uniform", d_param={"loc": 0, "scale": 1}
        )

        self.model_2d = gr.Model(
            functions=[
                gr.Function(lambda x: [x[0], x[1]], ["x", "y"], ["f", "g"], "test", 0)
            ],
            domain=domain_2d,
            density=gr.Density(
                marginals=marginals, copula=gr.CopulaIndependence(var_rand=["x"])
            ),
        )

        ## Correct results
        self.df_2d_nominal = pd.DataFrame(
            data={"x": [0.0], "y": [0.5], "f": [0.0], "g": [0.5]}
        )
        self.df_2d_grad = pd.DataFrame(
            data={"Df_Dx": [1.0], "Dg_Dx": [0.0], "Df_Dy": [0.0], "Dg_Dy": [1.0]}
        )
        self.df_2d_qe = pd.DataFrame(
            data={"x": [0.0], "y": [0.1], "f": [0.0], "g": [0.1]}
        )

    ## Test default evaluations

    def test_nominal(self):
        """Checks the nominal evaluation is accurate
        """
        df_res = gr.eval_nominal(self.model_2d)

        ## Accurate
        self.assertTrue(gr.df_equal(self.df_2d_nominal, df_res))

        ## Pass-through
        self.assertTrue(
            gr.df_equal(
                self.df_2d_nominal.drop(["f", "g"], axis=1),
                gr.eval_nominal(self.model_2d, skip=True),
            )
        )

    def test_grad_fd(self):
        """Checks the FD code
        """
        ## Accuracy
        df_grad = gr.eval_grad_fd(
            self.model_2d, df_base=self.df_2d_nominal, append=False
        )

        self.assertTrue(np.allclose(df_grad[self.df_2d_grad.columns], self.df_2d_grad))

        ## Subset
        df_grad_sub = gr.eval_grad_fd(
            self.model_2d, df_base=self.df_2d_nominal, var=["x"], append=False
        )

        self.assertTrue(set(df_grad_sub.columns) == set(["Df_Dx", "Dg_Dx"]))

        ## Flags
        md_test = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: x[0] + x[1] ** 2, var=2, out=1)
            >> gr.cp_marginals(x0={"dist": "norm", "loc": 0, "scale": 1})
        )
        df_base = pd.DataFrame(dict(x0=[0, 1], x1=[0, 1]))
        ## Multiple base points
        df_true = pd.DataFrame(dict(Dy0_Dx0=[1, 1], Dy0_Dx1=[0, 2]))

        df_rand = gr.eval_grad_fd(md_test, df_base=df_base, var="rand", append=False)
        self.assertTrue(gr.df_equal(df_true[["Dy0_Dx0"]], df_rand, close=True))

        df_det = gr.eval_grad_fd(md_test, df_base=df_base, var="det", append=False)
        self.assertTrue(gr.df_equal(df_true[["Dy0_Dx1"]], df_det, close=True))

    def test_conservative(self):
        ## Accuracy
        df_res = gr.eval_conservative(self.model_2d, quantiles=[0.1, 0.1])

        self.assertTrue(gr.df_equal(self.df_2d_qe, df_res, close=True))

        ## Repeat scalar value
        self.assertTrue(
            gr.df_equal(
                self.df_2d_qe,
                gr.eval_conservative(self.model_2d, quantiles=0.1),
                close=True,
            )
        )

        ## Pass-through
        self.assertTrue(
            gr.df_equal(
                self.df_2d_qe.drop(["f", "g"], axis=1),
                gr.eval_conservative(self.model_2d, quantiles=0.1, skip=True),
                close=True,
            )
        )


##################################################
class TestRandomSampling(unittest.TestCase):
    def setUp(self):
        self.md = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: x, var=1, out=1)
            >> gr.cp_marginals(x0={"dist": "uniform", "loc": 0, "scale": 1})
            >> gr.cp_copula_independence()
        )

        self.md_2d = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: x[0], var=2, out=1)
            >> gr.cp_marginals(
                x0={"dist": "uniform", "loc": 0, "scale": 1},
                x1={"dist": "uniform", "loc": 0, "scale": 1},
            )
            >> gr.cp_copula_independence()
        )

    def test_lhs(self):
        ## Accurate
        n = 2
        df_res = ev.eval_lhs(self.md_2d, n=n, df_det="nom", seed=101)

        np.random.seed(101)
        df_truth = pd.DataFrame(data=lhs(2, samples=n), columns=["x0", "x1"])
        df_truth["y0"] = df_truth["x0"]

        self.assertTrue(gr.df_equal(df_res, df_truth))

        ## Rounding
        df_round = ev.eval_lhs(self.md_2d, n=n + 0.1, df_det="nom", seed=101)

        self.assertTrue(gr.df_equal(df_round, df_truth))

        ## Pass-through
        df_pass = ev.eval_lhs(self.md_2d, n=n, skip=True, df_det="nom", seed=101)

        self.assertTrue(gr.df_equal(df_pass, df_truth[["x0", "x1"]]))

    def test_monte_carlo(self):
        ## Accurate
        n = 2
        df_res = gr.eval_monte_carlo(self.md, n=n, df_det="nom", seed=101)

        np.random.seed(101)
        df_truth = pd.DataFrame({"x0": np.random.random(n)})
        df_truth["y0"] = df_truth["x0"]

        self.assertTrue(gr.df_equal(df_res, df_truth))

        ## Rounding
        df_round = gr.eval_monte_carlo(self.md, n=n + 0.1, df_det="nom", seed=101)

        self.assertTrue(gr.df_equal(df_round, df_truth))

        ## Pass-through
        df_pass = gr.eval_monte_carlo(self.md, n=n, skip=True, df_det="nom", seed=101)

        self.assertTrue(gr.df_equal(df_pass[["x0"]], df_truth[["x0"]]))


##################################################
class TestRandom(unittest.TestCase):
    def setUp(self):
        self.md = models.make_test()

        self.md_mixed = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: x[0], var=3, out=1)
            >> gr.cp_bounds(x2=(0, 1))
            >> gr.cp_marginals(
                x0={"dist": "uniform", "loc": 0, "scale": 1},
                x1={"dist": "uniform", "loc": 0, "scale": 1},
            )
            >> gr.cp_copula_independence()
        )

    def test_monte_carlo(self):
        df_min = gr.eval_monte_carlo(self.md, df_det="nom")
        self.assertTrue(df_min.shape == (1, self.md.n_var + self.md.n_out))
        self.assertTrue(set(df_min.columns) == set(self.md.var + self.md.out))

        df_seeded = gr.eval_monte_carlo(self.md, df_det="nom", seed=101)
        df_piped = self.md >> gr.ev_monte_carlo(df_det="nom", seed=101)
        self.assertTrue(df_seeded.equals(df_piped))

        df_skip = gr.eval_monte_carlo(self.md, df_det="nom", skip=True)
        self.assertTrue(set(df_skip.columns) == set(self.md.var))

        df_noappend = gr.eval_monte_carlo(self.md, df_det="nom", append=False)
        self.assertTrue(set(df_noappend.columns) == set(self.md.out))

    def test_lhs(self):
        df_seeded = ev.eval_lhs(self.md, n=10, df_det="nom", seed=101)
        df_piped = self.md >> ev.ev_lhs(df_det="nom", n=10, seed=101)
        self.assertTrue(df_seeded.equals(df_piped))

        df_skip = ev.eval_lhs(self.md, n=1, df_det="nom", skip=True)
        self.assertTrue(set(df_skip.columns) == set(self.md.var))

        df_noappend = ev.eval_lhs(self.md, n=1, df_det="nom", append=False)
        self.assertTrue(set(df_noappend.columns) == set(self.md.out))

    def test_sinews(self):
        df_min = gr.eval_sinews(self.md, df_det="nom")
        self.assertTrue(
            set(df_min.columns)
            == set(self.md.var + self.md.out + ["sweep_var", "sweep_ind"])
        )
        self.assertTrue(df_min._plot_info["type"] == "sinew_outputs")

        df_seeded = gr.eval_sinews(self.md, df_det="nom", seed=101)
        df_piped = self.md >> gr.ev_sinews(df_det="nom", seed=101)
        self.assertTrue(df_seeded.equals(df_piped))

        df_skip = gr.eval_sinews(self.md, df_det="nom", skip=True)
        self.assertTrue(df_skip._plot_info["type"] == "sinew_inputs")

        df_mixed = gr.eval_sinews(self.md_mixed, df_det="swp")

    def test_hybrid(self):
        df_min = gr.eval_hybrid(self.md, df_det="nom")
        self.assertTrue(
            set(df_min.columns) == set(self.md.var + self.md.out + ["hybrid_var"])
        )
        self.assertTrue(df_min._meta["type"] == "eval_hybrid")

        df_seeded = gr.eval_hybrid(self.md, df_det="nom", seed=101)
        df_piped = self.md >> gr.ev_hybrid(df_det="nom", seed=101)
        self.assertTrue(df_seeded.equals(df_piped))

        df_total = gr.eval_hybrid(self.md, df_det="nom", plan="total")
        self.assertTrue(
            set(df_total.columns) == set(self.md.var + self.md.out + ["hybrid_var"])
        )
        self.assertTrue(df_total._meta["type"] == "eval_hybrid")

        df_skip = gr.eval_hybrid(self.md, df_det="nom", skip=True)
        self.assertTrue(set(df_skip.columns) == set(self.md.var + ["hybrid_var"]))

        ## Raises
        md_buckle = models.make_plate_buckle()
        with self.assertRaises(ValueError):
            gr.eval_hybrid(md_buckle, df_det="nom")


##################################################
class TestOpt(unittest.TestCase):
    def test_nls(self):
        ## Setup
        md_feat = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: x[0] * x[1] + x[2], var=3, out=1,)
            >> gr.cp_bounds(x0=[-1, +1], x2=[0, 0])
            >> gr.cp_marginals(x1=dict(dist="norm", loc=0, scale=1))
        )

        md_const = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: x[0], var=1, out=1)
            >> gr.cp_bounds(x0=(-1, +1))
        )

        df_response = md_feat >> gr.ev_df(
            df=gr.df_make(x0=0.1, x1=[-1, -0.5, +0, +0.5, +1], x2=0)
        )
        df_data = df_response[["x1", "y0"]]

        ## Model with features
        df_true = gr.df_make(x0=0.1)
        df_fit = md_feat >> gr.ev_nls(df_data=df_data, append=False)

        pd.testing.assert_frame_equal(
            df_fit,
            df_true,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )
        ## Fitting synonym
        md_feat_fit = df_data >> gr.ft_nls(md=md_feat, verbose=False)
        self.assertTrue(set(md_feat_fit.var) == set(["x1", "x2"]))

        ## Constant model
        df_const = gr.df_make(x0=0)
        df_fit = md_const >> gr.ev_nls(df_data=gr.df_make(y0=[-1, 0, +1]))

        pd.testing.assert_frame_equal(
            df_fit,
            df_const,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )

        ## Multiple restarts works
        df_multi = gr.eval_nls(md_feat, df_data=df_data, n_restart=2)
        self.assertTrue(df_multi.shape[0] == 2)

        ## Specified initial guess
        df_spec = gr.eval_nls(
            md_feat, df_data=df_data, df_init=gr.df_make(x0=0.5), append=False
        )
        pd.testing.assert_frame_equal(
            df_spec,
            df_true,
            check_exact=False,
            check_dtype=False,
            check_column_type=False,
        )
        # Raises if incorrect guess data
        with self.assertRaises(ValueError):
            gr.eval_nls(md_feat, df_data=df_data, df_init=gr.df_make(foo=0.5))

    def test_opt(self):
        md_bowl = (
            gr.Model("Constrained bowl")
            >> gr.cp_function(
                fun=lambda x: x[0] ** 2 + x[1] ** 2, var=["x", "y"], out=["f"],
            )
            >> gr.cp_function(
                fun=lambda x: (x[0] + x[1] + 1), var=["x", "y"], out=["g1"],
            )
            >> gr.cp_function(
                fun=lambda x: -(-x[0] + x[1] - np.sqrt(2 / 10)),
                var=["x", "y"],
                out=["g2"],
            )
            >> gr.cp_bounds(x=(-1, +1), y=(-1, +1),)
        )

        df_res = md_bowl >> gr.ev_min(out_min="f", out_geq=["g1"], out_leq=["g2"],)

        # Check result
        self.assertTrue(abs(df_res.x[0] + np.sqrt(1 / 20)) < 1e-6)
        self.assertTrue(abs(df_res.y[0] - np.sqrt(1 / 20)) < 1e-6)

        # Check errors for violated invariants
        with self.assertRaises(ValueError):
            gr.eval_min(md_bowl, out_min="FALSE")
        with self.assertRaises(ValueError):
            gr.eval_min(md_bowl, out_min="f", out_geq=["FALSE"])
        with self.assertRaises(ValueError):
            gr.eval_min(md_bowl, out_min="f", out_eq=["FALSE"])

        # Test multiple restarts
        df_multi = gr.eval_min(
            md_bowl, out_min="f", out_geq=["g1"], out_leq=["g2"], n_restart=2,
        )
        self.assertTrue(df_multi.shape[0] == 2)


## Run tests
if __name__ == "__main__":
    unittest.main()
