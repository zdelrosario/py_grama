import numpy as np
import pandas as pd
import unittest

from collections import OrderedDict as od
from context import core
from context import grama as gr
from context import models
from pyDOE import lhs

DF = gr.Intention()

##################################################
class TestEvalInvariants(unittest.TestCase):
    """Helper class for testing invariant errors for eval_* functions"""
    def __init__(self):
        """Setup necessary values"""
        self.md = (
            gr.Model()
            >> gr.cp_function(fun=lambda x: x, var=1, out=1, runtime=1)
        )
        self.md_var_det = self.md >> gr.cp_bounds(x1=(0, 1))
        self.df = pd.DataFrame(
            data={"x": [0.0], "y": [0.5]}
        )
        # declare tests
        self.type_tests = [(1,2), 2, [1, 8]]

    def md_arg(self, func, df_arg = "df", **kwargs):
        """Helper function for TypeErrors and ValueErrors for invalid
        Model arguments (eval_* functions).

        Args:
            func (func): eval function to test
            df_arg (str): name of DataFrame argument
            **kwargs: kwargs to pass"""

        ## Type test
        for wrong in self.type_tests:
            self.assertRaises(TypeError, func, wrong, **{df_arg:self.df}, **kwargs)

        ## No model.functions
        self.assertRaises(ValueError, func, gr.Model(), **{df_arg:self.df}, **kwargs)

    def df_arg(self, func, df_arg = "df", shortcut=False, acc_none="Never", **kwargs):
        """Helper function for testing for TypeErrors and ValueErrors for invalid
        DataFrame arguments (eval_* functions).

        Args:
            func (func): eval function to test
            df_arg (str): name of DataFrame argument
            shortcut (bool): if func has valid str shortcut for df arg
            acc_none (str or None): if func accepts None for df;
                "var_det": accepts None when model.n_var_det == 0;
                "always": always accepts None as input
        """
        ## General type tests
        for wrong in self.type_tests:
            self.assertRaises(TypeError, func, self.md, **{df_arg:wrong}, **kwargs)

        ## Str type test
        if shortcut:
            # wrong str shortcut test
            self.assertRaises(ValueError, func, self.md, **{df_arg:"a"}, **kwargs)
        else:
            # test any str for typeerror
            self.assertRaises(TypeError, func, self.md, **{df_arg:"nom"}, **kwargs)

        ## None checks
        if acc_none == "var_det" or acc_none == "never":
            # `None` check when model.n_var_det > 0
            self.assertRaises(TypeError, func, self.md_var_det, **{df_arg:None}, **kwargs)
        if acc_none == "never":
            # none not accepted under any condition, test when md.var_det==0
            self.assertRaises(TypeError, func, self.md, **{df_arg:None}, **kwargs)

    def df_arg_2(self, func, df_args,
                    shortcut=[False, False], acc_none=["Never", "Never"],
                    **kwargs):
        """Helper function for testing for TypeErrors and ValueErrors for
        DataFrame arguments in eval_* functions with two df inputs.

        Args:
            func (func): eval function to test
            df_arg1 (list(str)): name of DataFrame args
            shortcut (list(bool)): if func has valid str shortcut for either or
                both df args
            acc_none (list(str) or list(None)): if func accepts None for df args;
                "var_det": accepts None when model.n_var_det == 0;
                "always": always accepts None as input
                "never": never accepts None as input
            """
        ## General type tests for both dfs
        for wrong in self.type_tests:
            self.assertRaises(TypeError, func, self.md,
                                **{df_args[0]:wrong}, **{df_args[1]:self.df},
                                **kwargs)
            self.assertRaises(TypeError, func, self.md,
                                **{df_args[0]:self.df}, **{df_args[1]:wrong,
                                **kwargs})

        ## Str type test
        # df_args[0]
        if shortcut[0]:
            # wrong str shortcut test
            self.assertRaises(ValueError, func, self.md,
                                **{df_args[0]:"a"}, **{df_args[1]:self.df},
                                **kwargs)
        else:
            # test any str for typeerror
            self.assertRaises(TypeError, func, self.md,
                                **{df_args[0]:"nom"}, **{df_args[1]:self.df},
                                **kwargs)
        # df_args[1]
        if shortcut[1]:
            # wrong str shortcut test
            self.assertRaises(ValueError, func, self.md,
                                **{df_args[0]:self.df}, **{df_args[1]:"a"},
                                **kwargs)
        else:
            # test any str for typeerror
            self.assertRaises(TypeError, func, self.md,
                                **{df_args[0]:self.df}, **{df_args[1]:"nom"},
                                **kwargs)

        ## None checks
        if acc_none[0] == "var_det" or acc_none[0] is None:
            # `None` check when model.n_var_det > 0
            self.assertRaises(TypeError, func, self.md_var_det,
                                **{df_args[0]:None}, **{df_args[1]:self.df},
                                **kwargs)
        if acc_none[0] is None:
            # none not accepted under any condition, test when md.var_det==0
            self.assertRaises(TypeError, func, self.md,
                                **{df_args[0]:None}, **{df_args[1]:self.df},
                                **kwargs)
        # df_arg_2
        if acc_none[1] == "var_det" or acc_none[1] is None:
            # `None` check when model.n_var_det > 0
            self.assertRaises(TypeError, func, self.md_var_det,
                                **{df_args[0]:self.df}, **{df_args[1]:None},
                                **kwargs)
        if acc_none[1] is None:
            # none not accepted under any condition, test when md.var_det==0
            self.assertRaises(TypeError, func, self.md,
                                **{df_args[0]:self.df}, **{df_args[1]:None},
                                **kwargs)

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
                gr.Function(lambda x, y: [x, y], ["x", "y"], ["f", "g"], "test", 0)
            ],
            domain=domain_2d,
            density=gr.Density(
                marginals=marginals, copula=gr.CopulaIndependence(var_rand=["x"])
            ),
        )

        ## Invariant test class
        self.inv_test = TestEvalInvariants()

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

    def test_nominal(self):
        """Checks the nominal evaluation is accurate
        """
        df_res = gr.eval_nominal(self.model_2d)

        ## Accurate
        self.assertTrue(gr.df_equal(self.df_2d_nominal, df_res))

        ## Invariant checks
        self.inv_test.md_arg(gr.eval_nominal, df_arg="df_det")
        self.inv_test.df_arg(gr.eval_nominal, df_arg="df_det", shortcut=True, acc_none="var_det")

        ## Pass-through
        self.assertTrue(
            gr.df_equal(
                self.df_2d_nominal.drop(["f", "g"], axis=1),
                gr.eval_nominal(self.model_2d, skip=True),
            )
        )

    def test_linup(self):
        """Checks linear uncertainty propagation tool
        """
        # Setup
        model_3d = (
            gr.Model()
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(
                    y0=df.x0 + df.x1 + df.x2,
                    y1=df.x0 * gr.exp(df.x1 + df.x2),
                ),
                var=3,
                out=2
            )
            >> gr.cp_bounds(x0=(-1, +1))
            >> gr.cp_marginals(
                x1=gr.marg_mom("norm", mean=0, sd=1),
                x2=gr.marg_mom("uniform", mean=0, sd=1),
            )
            >> gr.cp_copula_gaussian(df_corr=gr.df_make(var1=["x1"], var2=["x2"], corr=[0.5]))
        )

        model_50 = (
            gr.Model()
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(
                    y0=df.x0 + df.x1 + df.x2,
                ),
                var=3,
                out=1,
            )
            >> gr.cp_bounds(x0=(-1, +1))
            >> gr.cp_marginals(
                x1=gr.marg_mom("norm", mean=0, sd=1),
                x2=gr.marg_mom("norm", mean=0, sd=1),
            )
            >> gr.cp_copula_gaussian(df_corr=gr.df_make(var1=["x1"], var2=["x2"], corr=[0.5]))
        )
        model_75 = (
            gr.Model()
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(
                    y0=df.x0 + df.x1 + df.x2,
                ),
                var=3,
                out=1,
            )
            >> gr.cp_bounds(x0=(-1, +1))
            >> gr.cp_marginals(
                x1=gr.marg_mom("norm", mean=0, sd=1),
                x2=gr.marg_mom("norm", mean=0, sd=1),
            )
            >> gr.cp_copula_gaussian(df_corr=gr.df_make(var1=["x1"], var2=["x2"], corr=[0.75]))
        )

        ## Test for accuracy
        df_res_50 = gr.eval_linup(model_50, df_base="nom", n=1e5, seed=101)
        self.assertTrue(abs(df_res_50["var"].values[0] - 3) / 3 <= 0.05)
        df_res_75 = gr.eval_linup(model_75, df_base="nom", n=1e5, seed=101)
        self.assertTrue(abs(df_res_75["var"].values[0] - 3.5) / 3.5 <= 0.05)

        ## Test for evaluation at base values
        df_base = gr.df_make(x0=0, x1=-1, x2=1)
        df_3d_base = gr.eval_linup(model_3d, df_base=df_base)
        self.assertTrue(gr.df_equal(
            pd.concat((df_base, df_base), axis=0).reset_index(drop=True),
            df_3d_base[df_base.columns],
        ))

        ## Test for outer product of base and outputs
        df_3d_outer = gr.eval_linup(model_3d, df_base=gr.df_make(x0=0, x1=0, x2=[0, 1]))
        self.assertTrue(df_3d_outer.shape[0] == 2 * 2)

        ## Test that sensitivity contributions sum to 1
        df_3d_nom = gr.eval_linup(model_3d, df_base=gr.df_make(x0=1, x1=0, x2=0), decomp=True)
        df_sum = (
            df_3d_nom
            >> gr.tf_group_by("out")
            >> gr.tf_summarize(s=gr.sum(DF.var_frac))
        )
        self.assertTrue(all(np.abs(df_sum.s - 1) <= 0.01))


    def test_grad_fd(self):
        """Checks the FD code
        """
        ## Accuracy
        df_grad = gr.eval_grad_fd(
            self.model_2d, df_base=self.df_2d_nominal, append=False
        )

        self.assertTrue(np.allclose(df_grad[self.df_2d_grad.columns], self.df_2d_grad))

        ## Invariant checks
        self.inv_test.md_arg(gr.eval_grad_fd, df_arg="df_base")
        self.inv_test.df_arg(gr.eval_grad_fd, df_arg="df_base")

        ## Subset
        df_grad_sub = gr.eval_grad_fd(
            self.model_2d, df_base=self.df_2d_nominal, var=["x"], append=False
        )

        self.assertTrue(set(df_grad_sub.columns) == set(["Df_Dx", "Dg_Dx"]))

        ## Flags
        md_test = (
            gr.Model()
            >> gr.cp_function(fun=lambda x0, x1: x0 + x1 ** 2, var=2, out=1)
            >> gr.cp_marginals(x0={"dist": "norm", "loc": 0, "scale": 1})
        )
        df_base = pd.DataFrame(dict(x0=[0, 1], x1=[0, 1]))
        ## Multiple base points
        df_true = pd.DataFrame(dict(Dy0_Dx0=[1, 1], Dy0_Dx1=[0, 2]))

        df_rand = gr.eval_grad_fd(md_test, df_base=df_base, var="rand", append=False)
        self.assertTrue(gr.df_equal(df_true[["Dy0_Dx0"]], df_rand, close=True))

        df_det = gr.eval_grad_fd(md_test, df_base=df_base, var="det", append=False)
        self.assertTrue(gr.df_equal(df_true[["Dy0_Dx1"]], df_det, close=True))
        ## Append base points
        df_append = gr.eval_grad_fd(md_test, df_base=df_base, var="det", append=True)
        self.assertTrue(gr.df_equal(df_base, df_append[md_test.var]))

    def test_conservative(self):
        ## Accuracy
        df_res = gr.eval_conservative(self.model_2d, quantiles=[0.1, 0.1])

        self.assertTrue(gr.df_equal(self.df_2d_qe, df_res, close=True))

        ## Invariant checks
        self.inv_test.md_arg(gr.eval_conservative, df_arg="df_det")
        self.inv_test.df_arg(gr.eval_conservative, df_arg="df_det",
                                shortcut=True, acc_none="var_det")

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
            >> gr.cp_function(fun=lambda x: x, var=1, out=1, runtime=1)
            >> gr.cp_marginals(x0={"dist": "uniform", "loc": 0, "scale": 1})
            >> gr.cp_copula_independence()
        )

        self.md_2d = (
            gr.Model()
            >> gr.cp_function(fun=lambda x0, x1: x0, var=2, out=1)
            >> gr.cp_marginals(
                x0={"dist": "uniform", "loc": 0, "scale": 1},
                x1={"dist": "uniform", "loc": 0, "scale": 1},
            )
            >> gr.cp_copula_independence()
        )

        self.md_mixed = (
            gr.Model()
            >> gr.cp_function(fun=lambda x0, x1: x0 + x1, var=2, out=1)
            >> gr.cp_bounds(x0=(-1, +1))
            >> gr.cp_marginals(
                x1={"dist": "uniform", "loc": 0, "scale": 1},
            )
            >> gr.cp_copula_independence()
        )

    def test_lhs(self):
        ## Accurate
        n = 2
        df_res = gr.eval_lhs(self.md_2d, n=n, df_det="nom", seed=101)

        np.random.seed(101)
        df_truth = pd.DataFrame(data=lhs(2, samples=n), columns=["x0", "x1"])
        df_truth["y0"] = df_truth["x0"]

        self.assertTrue(gr.df_equal(df_res, df_truth))

        ## Rounding
        df_round = gr.eval_lhs(self.md_2d, n=n + 0.1, df_det="nom", seed=101)

        self.assertTrue(gr.df_equal(df_round, df_truth))

        ## Pass-through
        df_pass = gr.eval_lhs(self.md_2d, n=n, skip=True, df_det="nom", seed=101)

        self.assertTrue(gr.df_equal(df_pass, df_truth[["x0", "x1"]]))

    def test_sample(self):
        ## Accurate
        n = 2
        df_res = gr.eval_sample(self.md, n=n, df_det="nom", seed=101)

        np.random.seed(101)
        df_truth = pd.DataFrame({"x0": np.random.random(n)})
        df_truth["y0"] = df_truth["x0"]

        self.assertTrue(gr.df_equal(df_res, df_truth))

        ## Rounding
        df_round = gr.eval_sample(self.md, n=n + 0.1, df_det="nom", seed=101)

        self.assertTrue(gr.df_equal(df_round, df_truth))

        ## Pass-through
        df_pass = gr.eval_sample(self.md, n=n, skip=True, df_det="nom", seed=101)

        self.assertTrue(gr.df_equal(df_pass[["x0"]], df_truth[["x0"]]))

        ## Optional observation index
        df_idx = gr.eval_sample(
            self.md_mixed,
            n=n,
            df_det=gr.df_make(x0=[-1, 0, 1]),
            seed=101,
            ind_comm="idx",
        )

        self.assertTrue(len(set(df_idx.idx)) == n)

        ## Common random numbers
        df_crn = gr.eval_sample(
            self.md_mixed,
            n=3,
            df_det=gr.df_make(x0=[0, 1]),
            seed=101,
        )
        self.assertTrue(len(set(df_crn.x1)) == 3)

        ## Non-common random numbers
        df_ncrn = gr.eval_sample(
            self.md_mixed,
            n=3,
            df_det=gr.df_make(x0=[0, 1]),
            seed=101,
            comm=False,
        )
        self.assertTrue(len(set(df_ncrn.x1)) == 6)


##################################################
class TestRandom(unittest.TestCase):
    def setUp(self):
        self.md = models.make_test()

        self.md_mixed = (
            gr.Model()
            >> gr.cp_function(fun=lambda x0, x1, x2: x0, var=3, out=1)
            >> gr.cp_bounds(x2=(0, 1))
            >> gr.cp_marginals(
                x0={"dist": "uniform", "loc": 0, "scale": 1},
                x1={"dist": "uniform", "loc": 0, "scale": 1},
            )
            >> gr.cp_copula_independence()
        )

        ## Invariant test class
        self.inv_test = TestEvalInvariants()

    def test_sample(self):
        # invariant checks
        self.inv_test.md_arg(gr.eval_sample, df_arg="df_det")
        self.inv_test.df_arg(gr.eval_sample, df_arg="df_det", shortcut=True,
                                    acc_none="var_det")

        # No `n` provided
        with self.assertRaises(ValueError):
            gr.eval_sample(self.md, df_det="nom")

        df_min = gr.eval_sample(self.md, n=1, df_det="nom")
        self.assertTrue(df_min.shape == (1, self.md.n_var + self.md.n_out))
        self.assertTrue(set(df_min.columns) == set(self.md.var + self.md.out))

        # Seed fixes runs
        df_seeded = gr.eval_sample(self.md, n=10, df_det="nom", seed=101)
        df_piped = self.md >> gr.ev_sample(df_det="nom", n=10, seed=101)
        self.assertTrue(df_seeded.equals(df_piped))

        df_skip = gr.eval_sample(self.md, n=1, df_det="nom", skip=True)
        self.assertTrue(set(df_skip.columns) == set(self.md.var))

        df_noappend = gr.eval_sample(self.md, n=1, df_det="nom", append=False)
        self.assertTrue(set(df_noappend.columns) == set(self.md.out))

    def test_lhs(self):
        # no inv test implemented, no need to test

        df_seeded = gr.eval_lhs(self.md, n=10, df_det="nom", seed=101)
        df_piped = self.md >> gr.ev_lhs(df_det="nom", n=10, seed=101)
        self.assertTrue(df_seeded.equals(df_piped))

        df_skip = gr.eval_lhs(self.md, n=1, df_det="nom", skip=True)
        self.assertTrue(set(df_skip.columns) == set(self.md.var))

        df_noappend = gr.eval_lhs(self.md, n=1, df_det="nom", append=False)
        self.assertTrue(set(df_noappend.columns) == set(self.md.out))

    def test_sinews(self):
        # invariant checks
        self.inv_test.md_arg(gr.eval_sinews, df_arg="df_det")
        self.inv_test.df_arg(gr.eval_sinews, df_arg="df_det", shortcut=True)

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
        # invariant checks
        self.inv_test.md_arg(gr.eval_hybrid, df_arg="df_det")
        self.inv_test.df_arg(gr.eval_hybrid, df_arg="df_det", shortcut=True)

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
    def setUp(self):
        ## Invariant test class
        self.inv_test = TestEvalInvariants()

    def test_nls(self):
        # invariant checks
        self.inv_test.md_arg(gr.eval_nls, df_arg="df_data")
        self.inv_test.df_arg_2(gr.eval_nls, df_args=["df_data", "df_init"],
            acc_none=["never", "always"])

        ## Setup
        md_feat = (
            gr.Model()
            >> gr.cp_function(fun=lambda x0, x1, x2: x0 * x1 + x2, var=3, out=1,)
            >> gr.cp_bounds(x0=[-1, +1], x2=[0, 0])
            >> gr.cp_marginals(x1=dict(dist="norm", loc=0, scale=1))
        )

        md_const = (
            gr.Model()
            >> gr.cp_function(fun=lambda x0: x0, var=1, out=1)
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
        # invariant checks
        self.inv_test.md_arg(gr.eval_min, df_arg="df_start")
        self.inv_test.df_arg(gr.eval_min, df_arg="df_start", acc_none="always")


        md_bowl = (
            gr.Model("Constrained bowl")
            >> gr.cp_function(
                fun=lambda x, y: x ** 2 + y ** 2, var=["x", "y"], out=["f"],
            )
            >> gr.cp_function(
                fun=lambda x, y: (x + y + 1), var=["x", "y"], out=["g1"],
            )
            >> gr.cp_function(
                fun=lambda x, y: -(-x + y - np.sqrt(2 / 10)),
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
