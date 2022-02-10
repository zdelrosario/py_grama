import unittest

from context import grama as gr
from context import data
import numpy as np
from numpy import array, pi
from grama.models import make_ishigami
from grama.fit import ft_gp

class TestEvalPND(unittest.TestCase):
    """Test implementation of eval_pnd
    """
    def test_eval_pnd(self):
        """ Test basic evaluation of pnd function
        """
        # Model to make dataset
        md_true = (
            gr.Model()
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(
                    y1=df.x1 * gr.cos(df.x2),
                    y2=df.x1 * gr.sin(df.x2),
                ),
                var=["x1", "x2"],
                out=["y1", "y2"],
            )
            >> gr.cp_marginals(
                x1=dict(dist="uniform", loc=0, scale=1),
                x2=dict(dist="uniform", loc=0, scale=pi/2),
            )
            >> gr.cp_copula_independence()
        )
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
        vars = df_train.columns.values
        input = md_fit.var
        outputs = [value for value in vars if value not in input]

        pr_scores, var_values = (
            md_fit
            >> gr.ev_pnd(
                df_train,
                df_test,
                sign = array([+1, +1]),
                seed = 101
            )
        )
        print(pr_scores)


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
        pr_scores, var_values = gr.approx_pnd(X_pred, X_cov, self.X_base, sign=np.array([-1, -1]), n=int(1e2))
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
            sign=np.array([+1, +1]),
            n=n,
            seed=101,
        )

        # Check for fixed level of accuracy
        self.assertTrue(np.abs(0.75 - pr_scores[0]) < 0.05)

        # Check that observed CI is compatible with true value
        pr_lo = pr_scores[0] - 2.58 * np.sqrt(var_values[0] / n) # 99% CI
        pr_hi = pr_scores[0] + 2.58 * np.sqrt(var_values[0] / n)
        self.assertTrue((pr_lo <= 0.75) and (0.75 <= pr_hi))
