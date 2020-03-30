import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest
import networkx as nx

from context import grama as gr
from context import models

## FD stepsize
h = 1e-8

## Core function tests
##################################################
class TestModel(unittest.TestCase):
    """Test implementation of model
    """

    def setUp(self):
        # Default model
        self.df_wrong = pd.DataFrame(data={"z": [0.0, 1.0]})

        # 2D identity model with permuted df inputs
        domain_2d = gr.Domain(bounds={"x": [-1.0, +1.0], "y": [0.0, 1.0]})
        marginals = {}
        marginals["x"] = gr.MarginalNamed(
            d_name="uniform", d_param={"loc": -1, "scale": 2}
        )
        marginals["y"] = gr.MarginalNamed(
            sign=-1, d_name="uniform", d_param={"loc": 0, "scale": 1}
        )

        self.model_2d = gr.Model(
            functions=[
                gr.Function(lambda x: [x[0], x[1]], ["x", "y"], ["x", "y"], "test", 0)
            ],
            domain=domain_2d,
            density=gr.Density(marginals=marginals),
        )
        self.df_2d = pd.DataFrame(data={"y": [0.0], "x": [+1.0]})
        self.res_2d = self.model_2d.evaluate_df(self.df_2d)

        self.df_median_in = pd.DataFrame({"x": [0.5], "y": [0.5]})
        self.df_median_out = pd.DataFrame({"x": [0.0], "y": [0.5]})

        self.model_3d = gr.Model(
            functions=[
                gr.Function(
                    lambda x: x[0] + x[1] + x[2], ["x", "y", "z"], ["f"], "test", 0
                )
            ],
            density=gr.Density(marginals=marginals),
        )

        ## Timing check
        self.model_slow = gr.Model(
            functions=[
                gr.Function(lambda x: x, ["x"], ["y"], "f0", 1),
                gr.Function(lambda x: x, ["x"], ["y"], "f1", 1),
            ]
        )

    def test_prints(self):
        ## Invoke printpretty
        self.model_3d.printpretty()

    def test_timings(self):
        ## Default is zero
        self.assertTrue(self.model_2d.runtime(1) == 0)

        ## Estimation accounts for both functions
        self.assertTrue(np.allclose(self.model_slow.runtime(1), 2))

        ## Fast function has empty message
        self.assertTrue(self.model_2d.runtime_message(self.df_2d) is None)

        ## Slow function returns string message
        msg = self.model_slow.runtime_message(pd.DataFrame({"x": [0]}))
        self.assertTrue(isinstance(msg, str))

    ## Basic functionality with default arguments

    def test_catch_input_mismatch(self):
        """Checks that proper exception is thrown if evaluate(df) passed a
        DataFrame without the proper columns.
        """
        self.assertRaises(ValueError, self.model_2d.evaluate_df, self.df_wrong)

    def test_var_outer(self):
        ## Test pass-throughs
        df_test = pd.DataFrame(dict(x0=[0]))
        md_no_rand = gr.Model() >> gr.cp_function(fun=lambda x: x, var=1, out=1)
        md_no_rand.var_outer(pd.DataFrame(), df_det="nom")

        md_no_det = md_no_rand >> gr.cp_marginals(
            x0={"dist": "uniform", "loc": 0, "scale": 1}
        )
        md_no_det.var_outer(df_test, df_det="nom")

        ## Test assertions
        with self.assertRaises(ValueError):
            self.model_3d.var_outer(self.df_2d)

        with self.assertRaises(ValueError):
            self.model_3d.var_outer(self.df_2d, df_det="foo")

        with self.assertRaises(ValueError):
            self.model_3d.var_outer(self.df_2d, df_det=self.df_2d)

    ## Test re-ordering issues

    def test_2d_output_names(self):
        """Checks that proper output names are assigned to resulting DataFrame
        """
        self.assertEqual(
            set(self.model_2d.evaluate_df(self.df_2d).columns), set(self.model_2d.out)
        )

    def test_2d_identity(self):
        """Checks that re-ordering of inputs handled properly
        """
        self.assertTrue(gr.df_equal(self.df_2d, self.res_2d.loc[:, self.df_2d.columns]))

    def test_quantile(self):
        """Checks that model.sample_quantile() evaluates correctly.
        """
        df_res = self.model_2d.density.pr2sample(self.df_median_in)

        self.assertTrue(gr.df_equal(df_res, self.df_median_out))

    def test_empty_functions(self):
        md = gr.Model() >> gr.cp_bounds(x=[-1, +1])
        with self.assertRaises(ValueError):
            gr.eval_nominal(md)

    ## Test sample transforms
    def test_transforms(self):
        ## Setup
        df_corr = pd.DataFrame(dict(var1=["x"], var2=["y"], corr=[0.5]))
        Sigma_h = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))

        md = (
            gr.Model()
            >> gr.cp_marginals(
                x=dict(dist="norm", loc=0, scale=1), y=dict(dist="norm", loc=0, scale=1)
            )
            >> gr.cp_copula_gaussian(df_corr=df_corr)
        )

        ## Transforms invariant
        z = np.array([0, 0])
        x = md.z2x(z)
        zp = md.x2z(x)

        self.assertTrue(np.all(z == zp))

        df_z = gr.df_make(x=0.0, y=0.0)
        df_x = md.norm2rand(df_z)
        df_zp = md.rand2norm(df_x)

        self.assertTrue(gr.df_equal(df_z, df_zp))

        ## Jacobian accurate
        dxdz_fd = np.zeros((2, 2))
        dxdz_fd[0, :] = (md.z2x(z + np.array([h, 0])) - md.z2x(z)) / h
        dxdz_fd[1, :] = (md.z2x(z + np.array([0, h])) - md.z2x(z)) / h
        dxdz_p = md.dxdz(z)

        self.assertTrue(np.allclose(dxdz_fd, dxdz_p))

    ## Test DAG construction

    def test_dag(self):
        md = (
            gr.Model()
            >> gr.cp_function(lambda x: x, var=1, out=1)
            >> gr.cp_function(lambda x: x[0] + x[1], var=["x0", "y0"], out=1)
        )

        G_true = nx.DiGraph()
        G_true.add_edge("(var)", "f0", label="{}".format({"x0"}))
        G_true.add_edge("f0", "(out)", label="{}".format({"y0"}))
        G_true.add_edge("(var)", "f1", label="{}".format({"x0"}))
        G_true.add_edge("f0", "f1", label="{}".format({"y0"}))
        G_true.add_edge("f1", "(out)", label="{}".format({"y1"}))

        self.assertTrue(
            nx.is_isomorphic(
                md.make_dag(),
                G_true,
                node_match=lambda u, v: u == v,
                edge_match=lambda u, v: u == v,
            )
        )


class TestEvalDf(unittest.TestCase):
    """Test implementation of eval_df()
    """

    def setUp(self):
        self.model = models.make_test()

    def test_catch_no_df(self):
        """Checks that eval_df() raises when no input df is given.
        """
        self.assertRaises(ValueError, gr.eval_df, self.model)


class TestMarginal(unittest.TestCase):
    def setUp(self):
        self.marginal_named = gr.MarginalNamed(
            d_name="norm", d_param={"loc": 0, "scale": 1}
        )

    def test_fcn(self):

        ## Invoke summary
        self.marginal_named.summary()

        ## Correct values for normal distribution
        self.assertTrue(self.marginal_named.l(0.5) == norm.pdf(0.5))
        self.assertTrue(self.marginal_named.p(0.5) == norm.cdf(0.5))
        self.assertTrue(self.marginal_named.q(0.5) == norm.ppf(0.5))


# --------------------------------------------------
class TestDomain(unittest.TestCase):
    def setUp(self):
        self.domain = gr.Domain(bounds={"x": (0, 1)})

    def test_blank(self):
        ## Test blank domain valid
        gr.Domain()

        ## Invoke summary
        self.domain.bound_summary("x")

        ## Invoke summary;
        self.assertTrue(self.domain.bound_summary("y").find("unbounded") > -1)


# --------------------------------------------------
class TestDensity(unittest.TestCase):
    def setUp(self):
        self.density = gr.Density(
            marginals=dict(
                x=gr.MarginalNamed(d_name="uniform", d_param={"loc": -1, "scale": 2}),
                y=gr.MarginalNamed(d_name="uniform", d_param={"loc": -1, "scale": 2}),
            ),
            copula=gr.CopulaGaussian(
                pd.DataFrame(dict(var1=["x"], var2=["y"], corr=[0.5]))
            ),
        )

    def test_copula_warning(self):
        md = gr.Model()

        with self.assertRaises(ValueError):
            md.density.sample()

    def test_CopulaIndependence(self):
        copula = gr.CopulaIndependence(var_rand=["x", "y"])
        df_res = copula.sample(seed=101)

        self.assertTrue(set(df_res.columns) == set(["x", "y"]))

        ## Transforms invariant
        z = np.array([0, 0])
        u = copula.z2u(z)
        zp = copula.u2z(u)

        self.assertTrue(np.all(z == zp))

        ## Jacobian accurate
        dudz_fd = np.zeros((2, 2))
        dudz_fd[0, :] = (copula.z2u(z + np.array([h, 0])) - copula.z2u(z)) / h
        dudz_fd[1, :] = (copula.z2u(z + np.array([0, h])) - copula.z2u(z)) / h
        dudz_p = copula.dudz(z)

        self.assertTrue(np.allclose(dudz_fd, dudz_p))

    def test_CopulaGaussian(self):
        df_corr = pd.DataFrame(dict(var1=["x"], var2=["y"], corr=[0.5]))
        Sigma_h = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
        copula = gr.CopulaGaussian(df_corr=df_corr)
        df_res = copula.sample(seed=101)

        self.assertTrue(np.isclose(copula.Sigma_h, Sigma_h).all)
        self.assertTrue(set(df_res.columns) == set(["x", "y"]))

        ## Test raises
        df_corr_invalid = pd.DataFrame(
            dict(var1=["x", "x"], var2=["y", "z"], corr=[0, 0])
        )

        with self.assertRaises(ValueError):
            gr.CopulaGaussian(df_corr=df_corr_invalid)

        ## Transforms invariant
        z = np.array([0, 0])
        u = copula.z2u(z)
        zp = copula.u2z(u)

        self.assertTrue(np.all(z == zp))

        ## Jacobian accurate
        dudz_fd = np.zeros((2, 2))
        dudz_fd[0, :] = (copula.z2u(z + np.array([h, 0])) - copula.z2u(z)) / h
        dudz_fd[1, :] = (copula.z2u(z + np.array([0, h])) - copula.z2u(z)) / h
        dudz_p = copula.dudz(z)

        self.assertTrue(np.allclose(dudz_fd, dudz_p))

    def test_conversion(self):
        df_pr_true = pd.DataFrame(dict(x=[0.5], y=[0.5]))
        df_sp_true = pd.DataFrame(dict(x=[0.0], y=[0.0]))

        df_pr_res = self.density.sample2pr(df_sp_true)
        df_sp_res = self.density.pr2sample(df_pr_true)

        self.assertTrue(gr.df_equal(df_pr_true, df_pr_res))
        self.assertTrue(gr.df_equal(df_sp_true, df_sp_res))

    def test_sampling(self):
        df_sample = self.density.sample(n=1, seed=101)

        self.assertTrue(set(df_sample.columns) == set(["x", "y"]))


# --------------------------------------------------
class TestFunction(unittest.TestCase):
    def setUp(self):
        self.fcn = gr.Function(lambda x: x, ["x"], ["x"], "test", 0)

        self.fcn_vec = gr.FunctionVectorized(lambda df: df, ["x"], ["x"], "test", 0)

        self.df = pd.DataFrame({"x": [0]})

        self.df_wrong = pd.DataFrame({"z": [0]})

    def test_function(self):
        fcn_copy = self.fcn.copy()

        self.assertTrue(self.fcn.var == fcn_copy.var)
        self.assertTrue(self.fcn.out == fcn_copy.out)
        self.assertTrue(self.fcn.name == fcn_copy.name)

        pd.testing.assert_frame_equal(
            self.df, self.fcn.eval(self.df), check_dtype=False
        )

        with self.assertRaises(ValueError):
            self.fcn.eval(self.df_wrong)

        ## Invoke summary
        self.fcn.summary()

    def test_function_vectorized(self):
        fcn_copy = self.fcn_vec.copy()

        self.assertTrue(self.fcn_vec.var == fcn_copy.var)
        self.assertTrue(self.fcn_vec.out == fcn_copy.out)
        self.assertTrue(self.fcn_vec.name == fcn_copy.name)

        pd.testing.assert_frame_equal(
            self.df, self.fcn_vec.eval(self.df), check_dtype=False
        )

    def test_function_model(self):
        md_base = gr.Model() >> gr.cp_function(
            fun=lambda x: x, var=1, out=1, name="name", runtime=1
        )

        ## Base constructor
        func = gr.FunctionModel(md_base)

        self.assertTrue(md_base.var == func.var)
        self.assertTrue(md_base.out == func.out)
        self.assertTrue(md_base.name == func.name)
        self.assertTrue(md_base.runtime(1) == func.runtime)

        ## Test copy
        func_copy = func.copy()

        self.assertTrue(func_copy.var == func.var)
        self.assertTrue(func_copy.out == func.out)
        self.assertTrue(func_copy.name == func.name)
        self.assertTrue(func_copy.runtime == func.runtime)


## Run tests
if __name__ == "__main__":
    unittest.main()
