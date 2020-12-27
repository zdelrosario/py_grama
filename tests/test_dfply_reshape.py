import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import data

X = gr.Intention()

##==============================================================================
## reshape test functions
##==============================================================================
def arrange_apply_helperfunc(df):
    df = df.sort_values("depth", ascending=False)
    df = df.head(5)
    return df


class TestReshape(unittest.TestCase):
    def setUp(self):
        self.df_elongated = data.df_diamonds >> gr.tf_gather(
            "variable", "value", add_id=True
        )

    def test_arrange(self):
        df = (
            data.df_diamonds.groupby("cut")
            .apply(arrange_apply_helperfunc)
            .reset_index(drop=True)
        )
        d = (
            data.df_diamonds
            >> gr.tf_group_by("cut")
            >> gr.tf_arrange("depth", ascending=False)
            >> gr.tf_head(5)
            >> gr.tf_ungroup()
        ).reset_index(drop=True)
        self.assertTrue(df.equals(d))

        d = (
            data.df_diamonds
            >> gr.tf_group_by("cut")
            >> gr.tf_arrange(X.depth, ascending=False)
            >> gr.tf_head(5)
            >> gr.tf_ungroup()
        ).reset_index(drop=True)
        assert df.equals(d)

        df = data.df_diamonds.sort_values(["cut", "price"], ascending=False) \
                             .reset_index(drop=True)
        d = data.df_diamonds >> gr.tf_arrange(gr.desc(X.cut), gr.desc(X.price))
        self.assertTrue(df.equals(d))

    def test_rename(self):
        df = data.df_diamonds.rename(
            columns={"cut": "Cut", "table": "Table", "carat": "Carat"}
        )
        d = data.df_diamonds >> gr.tf_rename(Cut=X.cut, Table=X.table, Carat="carat")
        self.assertTrue(df.equals(d))

    def test_gather(self):
        d = data.df_diamonds >> gr.tf_gather(
            "variable", "value", ["price", "depth", "x", "y", "z"]
        )

        variables = ["price", "depth", "x", "y", "z"]
        id_vars = [c for c in data.df_diamonds.columns if c not in variables]
        df = pd.melt(data.df_diamonds, id_vars, variables, "variable", "value")

        self.assertTrue(df.equals(d))

        d = data.df_diamonds >> gr.tf_gather("variable", "value")

        variables = data.df_diamonds.columns.tolist()
        id_vars = []
        df = pd.melt(data.df_diamonds, id_vars, variables, "variable", "value")

        self.assertTrue(df.equals(d))

        df = data.df_diamonds.copy()
        df["_ID"] = np.arange(df.shape[0])
        df = pd.melt(df, ["_ID"], variables, "variable", "value")

        self.assertTrue(df.equals(self.df_elongated))

    def test_spread(self):
        columns = self.df_elongated.columns.tolist()
        id_cols = ["_ID"]

        df = self.df_elongated.copy()
        df["temp_index"] = df["_ID"].values
        df = df.set_index("temp_index")
        spread_data = df[["variable", "value"]]

        spread_data = spread_data.pivot(columns="variable", values="value")
        converted_spread = spread_data.copy()

        columns_to_convert = [col for col in spread_data if col not in columns]
        converted_spread = gr.convert_type(converted_spread, columns_to_convert)

        df = df[["_ID"]].drop_duplicates()

        df_spread = df.merge(
            spread_data, left_index=True, right_index=True
        ).reset_index(drop=True)
        df_conv = df.merge(
            converted_spread, left_index=True, right_index=True
        ).reset_index(drop=True)

        d_spread = self.df_elongated >> gr.tf_spread("variable", "value")
        d_spread_conv = self.df_elongated >> gr.tf_spread(
            "variable", "value", convert=True
        )

        self.assertTrue(df_spread.equals(d_spread))
        self.assertTrue(df_conv.equals(d_spread_conv))

        ## Test fill
        df_base = gr.df_make(
            x=[1, 2, 3, 4, 5],
            y=["a", "b", "c", "a", "b"],
            idx=[0, 0, 0, 1, 1]
        )
        df_true = gr.df_make(
            a=[1, 4],
            b=[2, 5],
            c=[3, 0],
            idx=[0, 1]
        )
        df_res = df_base >> gr.tf_spread(X.y, X.x, fill=0)

        self.assertTrue(gr.df_equal(df_true, df_res, close=True))

    def test_separate(self):

        d = pd.DataFrame({"a": ["1-a-3", "1-b", "1-c-3-4", "9-d-1", "10"]})

        test1 = d >> gr.tf_separate(
            X.a,
            ["a1", "a2", "a3"],
            remove=True,
            convert=False,
            extra="merge",
            fill="right",
        )

        true1 = pd.DataFrame(
            {
                "a1": ["1", "1", "1", "9", "10"],
                "a2": ["a", "b", "c", "d", np.nan],
                "a3": ["3", np.nan, "3-4", "1", np.nan],
            }
        )
        self.assertTrue(true1.equals(test1))

        test2 = d >> gr.tf_separate(
            X.a,
            ["a1", "a2", "a3"],
            remove=True,
            convert=False,
            extra="merge",
            fill="left",
        )

        true2 = pd.DataFrame(
            {
                "a1": ["1", np.nan, "1", "9", np.nan],
                "a2": ["a", "1", "c", "d", np.nan],
                "a3": ["3", "b", "3-4", "1", "10"],
            }
        )
        self.assertTrue(true2.equals(test2))

        test3 = d >> gr.tf_separate(
            X.a,
            ["a1", "a2", "a3"],
            remove=True,
            convert=True,
            extra="merge",
            fill="right",
        )

        true3 = pd.DataFrame(
            {
                "a1": [1, 1, 1, 9, 10],
                "a2": ["a", "b", "c", "d", np.nan],
                "a3": ["3", np.nan, "3-4", "1", np.nan],
            }
        )
        self.assertTrue(true3.equals(test3))

        test4 = d >> gr.tf_separate(
            X.a,
            ["col1", "col2"],
            sep=[1, 3],
            remove=True,
            convert=False,
            extra="drop",
            fill="left",
        )

        true4 = pd.DataFrame(
            {"col1": ["1", "1", "1", "9", "1"], "col2": ["-a", "-b", "-c", "-d", "0"]}
        )
        self.assertTrue(true4.equals(test4))

        test5 = d >> gr.tf_separate(
            X.a,
            ["col1", "col2"],
            sep=[1, 3],
            remove=False,
            convert=False,
            extra="drop",
            fill="left",
        )

        true5 = pd.DataFrame(
            {
                "a": ["1-a-3", "1-b", "1-c-3-4", "9-d-1", "10"],
                "col1": ["1", "1", "1", "9", "1"],
                "col2": ["-a", "-b", "-c", "-d", "0"],
            }
        )
        self.assertTrue(true5.equals(test5))

        test6 = d >> gr.tf_separate(
            X.a,
            ["col1", "col2", "col3"],
            sep=[30],
            remove=True,
            convert=False,
            extra="drop",
            fill="left",
        )

        true6 = pd.DataFrame(
            {
                "col1": ["1-a-3", "1-b", "1-c-3-4", "9-d-1", "10"],
                "col2": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "col3": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        self.assertTrue(true6.equals(test6))

    def test_unite(self):
        d = pd.DataFrame(
            {"a": [1, 2, 3], "b": ["a", "b", "c"], "c": [True, False, np.nan]}
        )

        test1 = d >> gr.tf_unite(
            "united", X.a, "b", 2, remove=True, na_action="maintain"
        )
        true1 = pd.DataFrame({"united": ["1_a_True", "2_b_False", np.nan]})
        self.assertTrue(true1.equals(test1))

        test2 = d >> gr.tf_unite(
            "united", ["a", "b", "c"], remove=True, na_action="ignore", sep="*"
        )
        true2 = pd.DataFrame({"united": ["1*a*True", "2*b*False", "3*c"]})
        self.assertTrue(test2.equals(true2))

        test3 = d >> gr.tf_unite(
            "united", d.columns, remove=True, na_action="as_string"
        )
        true3 = pd.DataFrame({"united": ["1_a_True", "2_b_False", "3_c_nan"]})
        self.assertTrue(true3.equals(test3))

        test4 = d >> gr.tf_unite(
            "united", d.columns, remove=False, na_action="as_string"
        )
        true4 = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["a", "b", "c"],
                "c": [True, False, np.nan],
                "united": ["1_a_True", "2_b_False", "3_c_nan"],
            }
        )

        self.assertTrue(true4.equals(test4))

class TestNesting(unittest.TestCase):
    def setUp(self):
        pass

    def test_explode(self):
        df_base = gr.df_make(x=[1, 2], y=[[3, 4], [5, 6]])
        df_str = gr.df_make(x=[1, 2], y=[["3", "4"], ["5", "6"]])
        df_true = gr.df_make(
            x=[1, 1, 2, 2],
            y=[3, 4, 5, 6]
        )

        df_res = df_base >> gr.tf_explode(X.y)
        df_res_s = df_base >> gr.tf_explode(X.y, convert=True)

        self.assertTrue(gr.df_equal(df_true, df_res, close=True))
        self.assertTrue(gr.df_equal(df_true, df_res_s, close=True))
