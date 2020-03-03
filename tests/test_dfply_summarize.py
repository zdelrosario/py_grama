import numpy as np
import pandas as pd
import unittest

from context import grama as gr
from context import data

X = gr.Intention()

##==============================================================================
## summarization test functions
##==============================================================================


class TestSummarize(unittest.TestCase):
    def test_summarize(self):
        p = pd.DataFrame(
            {
                "price_mean": [data.df_diamonds.price.mean()],
                "price_std": [data.df_diamonds.price.std()],
            }
        )
        self.assertTrue(
            p.equals(
                data.df_diamonds
                >> gr.tf_summarize(price_mean=X.price.mean(), price_std=X.price.std())
            )
        )

        pcut = pd.DataFrame({"cut": ["Fair", "Good", "Ideal", "Premium", "Very Good"]})
        pcut["price_mean"] = [
            data.df_diamonds[data.df_diamonds.cut == c].price.mean()
            for c in pcut.cut.values
        ]
        pcut["price_std"] = [
            data.df_diamonds[data.df_diamonds.cut == c].price.std()
            for c in pcut.cut.values
        ]
        self.assertTrue(
            pcut.equals(
                data.df_diamonds
                >> gr.tf_group_by("cut")
                >> gr.tf_summarize(price_mean=X.price.mean(), price_std=X.price.std())
            )
        )

    def test_summarize_each(self):
        to_match = pd.DataFrame(
            {
                "price_mean": [np.mean(data.df_diamonds.price)],
                "price_var": [np.var(data.df_diamonds.price)],
                "depth_mean": [np.mean(data.df_diamonds.depth)],
                "depth_var": [np.var(data.df_diamonds.depth)],
            }
        )
        to_match = to_match[["price_mean", "price_var", "depth_mean", "depth_var"]]

        test1 = data.df_diamonds >> gr.tf_summarize_each([np.mean, np.var], X.price, 4)
        test2 = data.df_diamonds >> gr.tf_summarize_each(
            [np.mean, np.var], X.price, "depth"
        )
        self.assertTrue(to_match.equals(test1))
        self.assertTrue(to_match.equals(test2))

        group = pd.DataFrame({"cut": ["Fair", "Good", "Ideal", "Premium", "Very Good"]})
        group["price_mean"] = [
            np.mean(data.df_diamonds[data.df_diamonds.cut == c].price)
            for c in group.cut.values
        ]
        group["price_var"] = [
            np.var(data.df_diamonds[data.df_diamonds.cut == c].price)
            for c in group.cut.values
        ]
        group["depth_mean"] = [
            np.mean(data.df_diamonds[data.df_diamonds.cut == c].depth)
            for c in group.cut.values
        ]
        group["depth_var"] = [
            np.var(data.df_diamonds[data.df_diamonds.cut == c].depth)
            for c in group.cut.values
        ]

        group = group[["cut", "price_mean", "price_var", "depth_mean", "depth_var"]]

        test1 = (
            data.df_diamonds
            >> gr.tf_group_by(X.cut)
            >> gr.tf_summarize_each([np.mean, np.var], X.price, 4)
        )
        test2 = (
            data.df_diamonds
            >> gr.tf_group_by("cut")
            >> gr.tf_summarize_each([np.mean, np.var], X.price, "depth")
        )

        self.assertTrue(group.equals(test1))
        self.assertTrue(group.equals(test2))
