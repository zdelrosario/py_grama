import unittest
import io
import sys

from context import grama as gr

DF = gr.Intention()

## Test support points
##################################################
class TestPolyridge(unittest.TestCase):
    def setUp(self):
        pass

    def test_fit_polyridge(self):
        """Test the functionality and correctness of ft_polyridge()
        """
        df_test = (
            gr.df_make(x=range(10))
            >> gr.tf_outer(gr.df_make(y=range(10)))
            >> gr.tf_outer(gr.df_make(z=range(10)))
            >> gr.tf_mutate(f=DF.x - DF.y)
        )

        md = gr.fit_polyridge(df_test, out="f", n_degree=1, n_dim=1)

        df1 = gr.eval_df(md, df=gr.df_make(x=[2,1], y=[1], z=[0]))
        df2 = gr.df_make(x=[2,1], y=[1], z=[0], f_mean=[1, 0])

        self.assertTrue(gr.df_equal(
            df1,
            df2,
            close=True,
        ))

    def test_tran_polyridge(self):
        """Test the functionality and correctness of tran_polyridge()
        """
        ## Setup
        df_test = (
            gr.df_make(x=range(10))
            >> gr.tf_outer(gr.df_make(y=range(10)))
            >> gr.tf_outer(gr.df_make(z=range(10)))
            >> gr.tf_mutate(f=DF.x - DF.y)
        )

        ## Assertions
        # No `out` column
        with self.assertRaises(ValueError):
            gr.tran_polyridge(df_test)
        # Unrecognized `out` column
        with self.assertRaises(ValueError):
            gr.tran_polyridge(df_test, out="foo")
        # Unrecognized `var` column(s)
        with self.assertRaises(ValueError):
            gr.tran_polyridge(df_test, var=["foo", "bar"])
        # Invalid degree
        with self.assertRaises(ValueError):
            gr.tran_polyridge(df_test, out="f", n_degree=1, n_dim=2)

        ## Correctness
        df_res = (
            df_test
            >> gr.tf_polyridge(
                out="f",
                n_dim=1,
                n_degree=1,
            )
        )
        df_true = gr.df_make(x=1/gr.sqrt(2), y=-1/gr.sqrt(2), z=0)

        self.assertTrue(gr.df_equal(df_res, df_true, close=True))

        ## Higher-dimensional case runs without error
        df_higher = (
            gr.df_grid(
                x=range(10),
                y=range(10),
                z=range(10),
            )
            >> gr.tf_mutate(f=DF.x + DF.y + DF.z)
        )
        gr.tran_polyridge(df_higher, out="f", n_degree=2, n_dim=2)

        ## Fitting seed runs without error
        gr.fit_polyridge(df_higher, out="f", n_degree=1, n_dim=1, seed=101)

        ## Seed stabilizes results
        df_res1 = gr.tran_polyridge(df_higher, out="f", n_degree=1, n_dim=1, seed=101)
        df_res2 = gr.tran_polyridge(df_higher, out="f", n_degree=1, n_dim=1, seed=101)

        self.assertTrue(gr.df_equal(df_res1, df_res2, close=True))

## Run tests
if __name__ == '__main__':
    import xmlrunner

    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output='test-reports'),
        failfast=False,
        buffer=False,
        catchbreak=False,
    )
