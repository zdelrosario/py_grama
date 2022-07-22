import unittest

from context import grama as gr
from context import data
from test_evals import TestEvalInvariants
import numpy as np
from numpy import array, pi
from grama.models import make_pareto_random
from grama.fit import ft_gp

class TestEvalPND(unittest.TestCase):
    """Test implementation of eval_pnd
    """
    def setUp(self):
        ## Invariant test class
        self.inv_test = TestEvalInvariants()

    def test_eval_pnd(self):
        """ Test basic evaluation of pnd function
        """
        # Invariant checks
        self.inv_test.md_arg(
            gr.eval_pnd, 
            df_arg="df_train",
            df_test = self.inv_test.df,
            signs = {"y1":1, "y2":1},
            seed = 101)
        self.inv_test.df_arg_2(
            gr.eval_pnd, 
            df_args=["df_train", "df_test"],
            signs = {"y1":1, "y2":1},
            seed = 101)

        # Model to make Dataset
        md_true = make_pareto_random()
        # Create dataframe
        df_data = (
            md_true
            >> gr.ev_sample(n=2e3, seed=101, df_det="nom")
        )
        ## Select training set
        df_train = (
            df_data
            >> gr.tf_sample(n=10)
        )
        ## select test set
        df_test = (
            df_data
                >> gr.tf_anti_join(
                    df_train,
                    by=["x1", "x2"],
                )
                >> gr.tf_sample(n=200)
        )

        # Create fitted model
        md_fit = (
            df_train
            >> ft_gp(
                var=["x1", "x2"],
                out=["y1", "y2"],
            )
        )

        # Call eval_pnd
        df_pnd = (
            md_fit
            >> gr.ev_pnd(
                df_train,
                df_test,
                signs = {"y1":1, "y2":1},
                seed = 101
            )
        )

        # Test for correctness by shape
        self.assertTrue(len(df_pnd) == df_test.shape[0])
        # Test for correctness by # of outputs
        self.assertTrue(len(df_pnd.columns.values) == len(df_test.columns.values) + 2)

    def test_eval_3D(self):
        """ Test 3D case for eval_pnd()
        """
        # Model to make Dataset
        md_true = make_pareto_random(twoDim=False)

        # Create dataframe
        df_data = (
            md_true
            >> gr.ev_sample(n=2e3, seed=101, df_det="nom")
        )
        ## Select training set
        df_train = (
            df_data
            >> gr.tf_sample(n=10)
        )
        ## select test set
        df_test = (
            df_data
                >> gr.tf_anti_join(
                    df_train,
                    by=["x1", "x2"],
                )
                >> gr.tf_sample(n=200)
        )

        # Create fitted model
        md_fit = (
            df_train
            >> ft_gp(
                var=["x1", "x2", "x3"],
                out=["y1", "y2", "y3"],
            )
        )

        # Call eval_pnd
        df_pnd = (
            md_fit
            >> gr.ev_pnd(
                df_train,
                df_test,
                signs = {"y1":1, "y2":1,"y3":1},
                seed = 101
            )
        )

        # Test for correctness by shape
        self.assertTrue(len(df_pnd) == df_test.shape[0])
        # Test for correctness by # of outputs
        self.assertTrue(len(df_pnd.columns.values) == len(df_test.columns.values) + 2)

    def test_eval_append(self):
        """ Test append parameter on eval_pnd()
        """
        # Model to make Dataset
        md_true = make_pareto_random(twoDim=False)

        # Create dataframe
        df_data = (
            md_true
            >> gr.ev_sample(n=2e3, seed=101, df_det="nom")
        )
        ## Select training set
        df_train = (
            df_data
            >> gr.tf_sample(n=10)
        )
        ## select test set
        df_test = (
            df_data
                >> gr.tf_anti_join(
                    df_train,
                    by=["x1", "x2"],
                )
                >> gr.tf_sample(n=200)
        )

        # Create fitted model
        md_fit = (
            df_train
            >> ft_gp(
                var=["x1", "x2", "x3"],
                out=["y1", "y2", "y3"],
            )
        )

        # Call eval_pnd
        df_pnd = (
            md_fit
            >> gr.ev_pnd(
                df_train,
                df_test,
                signs = {"y1":1, "y2":1,"y3":1},
                seed = 101,
                append = False
            )
        )

        # Test for correctness by shape
        self.assertTrue(len(df_pnd) == df_test.shape[0])
        # Test for correctness by # of outputs
        self.assertTrue(len(df_pnd.columns.values) == 2)


    def test_eval_input_subsets(self):
        """ Test inputs are subsets of the provided DataFrames for eval_pnd()
        """
        # Model to make Dataset
        md_true = make_pareto_random(twoDim=False)

        # Create dataframe
        df_data = (
            md_true
            >> gr.ev_sample(n=2e3, seed=101, df_det="nom")
        )
        ## Select training set
        df_train = (
            df_data
            >> gr.tf_sample(n=10)
        )
        ## select test set
        df_test = (
            df_data
                >> gr.tf_anti_join(
                    df_train,
                    by=["x1", "x2"],
                )
                >> gr.tf_sample(n=200)
        )

        # Create fitted model
        md_fit = (
            df_train
            >> ft_gp(
                var=["x1", "x2", "x3"],
                out=["y1", "y2", "y3"],
            )
        )

        # Call eval_pnd w/ only "y1" and "y2"
        df_pnd = (
            md_fit
            >> gr.ev_pnd(
                df_train,
                df_test,
                signs = {"y1":1, "y2":1},
                seed = 101
            )
        )

        ### how to imply x1 and x2 from y1 and y2?

        # Test for correctness by shape
        self.assertTrue(len(df_pnd) == df_test.shape[0])
        # Test for correctness by # of outputs
        self.assertTrue(len(df_pnd.columns.values) == len(df_test.columns.values) + 2)

    def test_eval_faulty_inputs(self):
        """ Test faulty inputs to eval_pnd
        """
        # Model to make Dataset
        md_true = make_pareto_random()
        # Create dataframe
        df_data = (
            md_true
            >> gr.ev_sample(n=2e3, seed=101, df_det="nom")
        )
        ## Select training set
        df_train = (
            df_data
            >> gr.tf_sample(n=10)
        )
        ## select test set
        df_test = (
            df_data
                >> gr.tf_anti_join(
                    df_train,
                    by=["x1", "x2"],
                )
                >> gr.tf_sample(n=200)
        )

        # Create fitted model
        md_fit = (
            df_train
            >> ft_gp(
                var=["x1", "x2"],
                out=["y1", "y2"],
            )
        )

        # Call eval_pnd
        with self.assertRaises(ValueError):
            df_pnd = (
                md_fit
                >> gr.ev_pnd(
                    df_train,
                    df_test,
                    signs = {"y":1, "y2":1},
                    seed = 101
                )
            )


class TestUtilities(unittest.TestCase):

    def setUp(self):
        self.X_base = np.random.multivariate_normal(
            mean=[0, 0],
            cov=np.eye(2),
            size=50,
        )
        # Compute the pareto points
        self.idx_pareto = gr.pareto_min_rel(self.X_base)

    def test_pareto_min_rel(self):
        # Relative mode
        X_base = np.array([
            [1, 0],
            [0, 1],
        ])
        X_test = np.array([
            [0, 0],
            [1, 1],
        ])

        b_true = np.array([1, 0], dtype=bool)

        # Check accuracy
        self.assertTrue(
            np.all(gr.pareto_min_rel(X_test, X_base=X_base) == b_true)
        )

        # For Pareto frontier
        X_single = np.array([
            [2, 0],
            [1, 1],
            [0, 2],
            [2, 1],
            [2, 2],
            [1, 2],
        ])
        b_single = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        # Check accuracy
        res = gr.pareto_min_rel(X_single)
        self.assertTrue(
            np.all(res == b_single)
        )

    def test_fundamentals(self):
        # Test bare functionality
        Sigma = gr.make_proposal_sigma(self.X_base, self.idx_pareto, np.eye(2))
        X_sample = gr.rprop(10, Sigma, self.X_base[self.idx_pareto, :])
        d_sample = gr.dprop(X_sample, Sigma, self.X_base[self.idx_pareto, :])

    def test_pnd_order(self):
        """Test the PND MIS approximation for simple rank-ordering
        """
        # Against the common base-points, the following should be trivial to distinguish
        X_pred = np.array([
            [-1, -1],
            [ 0,  0],
            [+1, +1],
        ])
        X_cov = [np.eye(2)] * 3
        # A very coarse sample will suffice
        pr_scores, var_values = gr.approx_pnd(X_pred, X_cov, self.X_base, signs=np.array([-1, -1]), n=int(1e2))
        # Check basic ordering correctness
        self.assertTrue(pr_scores[0] > pr_scores[1])
        self.assertTrue(pr_scores[1] > pr_scores[2])

    def test_pnd_orthant(self):
        """Test the PND MIS approximation for accuracy with orthant tests
        """
        ## A single isotonic gaussian at the origin has a 75% chance of landing
        ## outside the negative orthant; we can use this as a numerical check of
        ## the PND MIS approximation
        X_pred = np.array([[0, 0]])
        X_cov = [np.eye(2) * 1e-1]
        X_base = np.array([
            [   -3,     0],
            [   -2,     0],
            [   -1,     0],
            [ -0.5,     0],
            [ -0.1,     0],
            [    0,     0],
            [    0,  -0.1],
            [    0,    -1],
            [    0,    -2],
            [    0,    -3],
        ])

        n = int(1e4)
        pr_scores, var_values = gr.approx_pnd(
            X_pred,
            X_cov,
            X_base,
            signs=np.array([+1, +1]),
            n=n,
            seed=101,
        )

        # Check for fixed level of accuracy
        self.assertTrue(np.abs(0.75 - pr_scores[0]) < 0.05)

        # Check that observed CI is compatible with true value
        pr_lo = pr_scores[0] - 2.58 * np.sqrt(var_values[0] / n) # 99% CI
        pr_hi = pr_scores[0] + 2.58 * np.sqrt(var_values[0] / n)
        self.assertTrue((pr_lo <= 0.75) and (0.75 <= pr_hi))

## Run tests
if __name__ == "__main__":
    unittest.main()