import unittest

from context import grama as gr
from context import data
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
            >> gr.ev_monte_carlo(n=2e3, seed=101, df_det="nom")
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

        pr_scores, var_values = (
            md_fit
            >> gr.ev_pnd(
                df_train,
                df_test,
                signs = array([+1, +1]),
                seed = 101
            )
        )

        # df_pnd = (
        #     df_pred
        #     >> gr.tf_mutate(
        #         pnd=pr_scores,
        #         pnd_sd=gr.sqrt(var_values),
        #     )
        # )

        # )
        # ## Compute predictions and predicted uncertainties
        # df_pred = (
        #     df_test
        #     >> gr.tf_md(md=md_fit)
        # )
        # ## Reshape data for algorithm
        # X_pred = df_pred[["y1_mean", "y2_mean"]].values # Predicted response values
        # X_sig = df_pred[["y1_sd", "y2_sd"]].values      # Predictive uncertainties
        # X_train = df_train[["y1", "y2"]].values         # Training response values
        # # PND expects covariance matrices; if we assume that
        # # the predictive covariance structures are independent,
        # # we can form these from diagonal matrices
        # X_cov = np.zeros((X_sig.shape[0], 2, 2))
        # for i in range(X_sig.shape[0]):
        #     X_cov[i, 0, 0] = X_sig[i, 0]
        #     X_cov[i, 1, 1] = X_sig[i, 1]
        #
        # ## Apply PND IS algorithm
        # pr_scores, var_values = approx_pnd(
        #     X_pred,
        #     X_cov,
        #     X_train,
        #     sign=np.array([+1, +1]),
        #     seed=101,
        # )
