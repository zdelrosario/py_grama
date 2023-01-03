import numpy as np
import pandas as pd
from scipy.stats import norm
import unittest

from context import grama as gr
from context import data

X = gr.Intention()

##==============================================================================
## select and drop test functions
##==============================================================================

#       0     1      2     3       4      5      6      7     8     9
#   carat    cut color clarity  depth  table  price     x     y     z
#    0.23  Ideal     E     SI2   61.5   55.0    326  3.95  3.98  2.43


class TestSelect(unittest.TestCase):
    def test_select(self):
        df = data.df_diamonds[["carat", "cut", "price"]]
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select("carat", "cut", "price"))
        )
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_select(0, 1, 6)))
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_select(0, 1, "price")))
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select([0, X.cut], X.price))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select(X.carat, X["cut"], X.price))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select(X[["carat", "cut", "price"]]))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select(X[["carat", "cut"]], X.price))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select(X.iloc[:, [0, 1, 6]]))
        )
        self.assertTrue(
            df.equals(
                data.df_diamonds >> gr.tf_select([X.loc[:, ["carat", "cut", "price"]]])
            )
        )

    def test_select_inversion(self):
        df = data.df_diamonds.iloc[:, 3:]
        d = data.df_diamonds >> gr.tf_select(~X.carat, ~X.cut, ~X.color)
        self.assertTrue(df.equals(d))

    def test_drop(self):
        df = data.df_diamonds.drop(["carat", "cut", "price"], axis=1)
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop("carat", "cut", "price"))
        )
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_drop(0, 1, 6)))
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_drop(0, 1, "price")))
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_drop([0, X.cut], X.price)))
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop(X.carat, X["cut"], X.price))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop(X[["carat", "cut", "price"]]))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop(X[["carat", "cut"]], X.price))
        )
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_drop(X.iloc[:, [0, 1, 6]])))
        self.assertTrue(
            df.equals(
                data.df_diamonds >> gr.tf_drop([X.loc[:, ["carat", "cut", "price"]]])
            )
        )

    def test_select_containing(self):
        df = data.df_diamonds[["carat", "cut", "color", "clarity", "price"]]
        assert df.equals(data.df_diamonds >> gr.tf_select(gr.contains("c")))

    def test_drop_containing(self):
        df = data.df_diamonds[["depth", "table", "x", "y", "z"]]
        assert df.equals(data.df_diamonds >> gr.tf_drop(gr.contains("c")))

    def test_select_matches(self):
        df = data.df_diamonds[["carat", "cut", "color", "clarity", "price"]]
        assert df.equals(data.df_diamonds >> gr.tf_select(gr.matches("^c[auol]|pri")))

    def test_drop_matches(self):
        df = data.df_diamonds[["depth", "table", "x", "y", "z"]]
        assert df.equals(data.df_diamonds >> gr.tf_drop(gr.matches("^c[auol]|p.i")))

    def test_select_startswith(self):
        df = data.df_diamonds[["carat", "cut", "color", "clarity"]]
        assert df.equals(data.df_diamonds >> gr.tf_select(gr.starts_with("c")))

    def test_drop_startswith(self):
        df = data.df_diamonds[["depth", "table", "price", "x", "y", "z"]]
        assert df.equals(data.df_diamonds >> gr.tf_drop(gr.starts_with("c")))

    def test_select_endswith(self):
        df = data.df_diamonds[["table", "price"]]
        assert df.equals(data.df_diamonds >> gr.tf_select(gr.ends_with("e")))

    def test_drop_endswith(self):
        df = data.df_diamonds.drop("z", axis=1)
        assert df.equals(data.df_diamonds >> gr.tf_drop(gr.ends_with("z")))

    def test_select_between(self):
        df = data.df_diamonds[["cut", "color", "clarity"]]
        self.assertTrue(
            df.equals(
                data.df_diamonds >> gr.tf_select(gr.columns_between(X.cut, X.clarity))
            )
        )
        self.assertTrue(
            df.equals(
                data.df_diamonds >> gr.tf_select(gr.columns_between("cut", "clarity"))
            )
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select(gr.columns_between(1, 3)))
        )

        df = data.df_diamonds[["x", "y", "z"]]
        assert df.equals(data.df_diamonds >> gr.tf_select(gr.columns_between("x", 20)))

    def test_drop_between(self):
        df = data.df_diamonds[["carat", "z"]]
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_between("cut", "y")))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_between(X.cut, 8)))
        )

        df = data.df_diamonds[["carat", "cut"]]
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_between(X.color, 20)))
        )

    def test_select_from(self):
        df = data.df_diamonds[["x", "y", "z"]]
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select(gr.columns_from("x")))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select(gr.columns_from(X.x)))
        )
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_select(gr.columns_from(7))))
        self.assertTrue(
            data.df_diamonds[[]].equals(
                data.df_diamonds >> gr.tf_select(gr.columns_from(100))
            )
        )

    def test_drop_from(self):
        df = data.df_diamonds[["carat", "cut"]]
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_from("color")))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_from(X.color)))
        )
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_from(2))))
        self.assertTrue(
            data.df_diamonds[[]].equals(
                data.df_diamonds >> gr.tf_drop(gr.columns_from(0))
            )
        )

    def test_select_to(self):
        df = data.df_diamonds[["carat", "cut"]]
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select(gr.columns_to("color")))
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_select(gr.columns_to(X.color)))
        )
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_select(gr.columns_to(2))))

    def test_drop_to(self):
        df = data.df_diamonds[["x", "y", "z"]]
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_to("x"))))
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_to(X.x))))
        self.assertTrue(df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_to(7))))

    def select_through(self):
        df = data.df_diamonds[["carat", "cut", "color"]]
        self.assertTrue(
            df.equals(
                data.df_diamonds >> gr.tf_select(gr.columns_to("color", inclusive=True))
            )
        )
        self.assertTrue(
            df.equals(
                data.df_diamonds >> gr.tf_select(gr.columns_to(X.color, inclusive=True))
            )
        )
        self.assertTrue(
            df.equals(
                data.df_diamonds >> gr.tf_select(gr.columns_to(2, inclusive=True))
            )
        )

    def drop_through(self):
        df = data.df_diamonds[["y", "z"]]
        self.assertTrue(
            df.equals(
                data.df_diamonds >> gr.tf_drop(gr.columns_to("x", inclusive=True))
            )
        )
        self.assertTrue(
            df.equals(
                data.df_diamonds >> gr.tf_drop(gr.columns_to(X.x, inclusive=True))
            )
        )
        self.assertTrue(
            df.equals(data.df_diamonds >> gr.tf_drop(gr.columns_to(7, inclusive=True)))
        )

    def test_select_if(self):
        # test 1: manually build data.df_diamonds subset where columns are numeric and
        # mean is greater than 3
        cols = list()
        for col in data.df_diamonds:
            try:
                if mean(data.df_diamonds[col]) > 3:
                    cols.append(col)
            except:
                pass
        df_if = data.df_diamonds[cols]
        self.assertTrue(
            df_if.equals(data.df_diamonds >> gr.tf_select_if(lambda col: mean(col) > 3))
        )
        # test 2: use and
        cols = list()
        for col in data.df_diamonds:
            try:
                if mean(data.df_diamonds[col]) > 3 and max(data.df_diamonds[col]) < 50:
                    cols.append(col)
            except:
                pass
        df_if = data.df_diamonds[cols]
        self.assertTrue(
            df_if.equals(
                data.df_diamonds
                >> gr.tf_select_if(lambda col: mean(col) > 3 and max(col) < 50)
            )
        )
        # test 3: use or
        cols = list()
        for col in data.df_diamonds:
            try:
                if mean(data.df_diamonds[col]) > 3 or max(data.df_diamonds[col]) < 6:
                    cols.append(col)
            except:
                pass
        df_if = data.df_diamonds[cols]
        self.assertTrue(
            df_if.equals(
                data.df_diamonds
                >> gr.tf_select_if(lambda col: mean(col) > 3 or max(col) < 6)
            )
        )
        # test 4: string operations - contain a specific string
        cols = list()
        for col in data.df_diamonds:
            try:
                if any(data.df_diamonds[col].str.contains("Ideal")):
                    cols.append(col)
            except:
                pass
        df_if = data.df_diamonds[cols]
        self.assertTrue(
            df_if.equals(
                data.df_diamonds
                >> gr.tf_select_if(lambda col: any(col.str.contains("Ideal")))
            )
        )
        # test 5: get any text columns
        # uses the special '.' regex symbol to find any text value
        cols = list()
        for col in data.df_diamonds:
            try:
                if any(data.df_diamonds[col].str.contains(".")):
                    cols.append(col)
            except:
                pass
        df_if = data.df_diamonds[cols]
        self.assertTrue(
            df_if.equals(
                data.df_diamonds
                >> gr.tf_select_if(lambda col: any(col.str.contains(".")))
            )
        )
        # test 6: is_numeric helper
        self.assertEqual(
            {"carat", "depth", "table", "price", "x", "y", "z"},
            set(
                (
                    data.df_diamonds
                    >> gr.tf_select_if(gr.is_numeric)
                ).columns
            )
        )

    def test_drop_if(self):
        # test 1: returns a dataframe where any column does not have a mean greater than 3
        # this means numeric columns with mean less than 3, and also any non-numeric column
        # (since it does not have a mean)
        cols = list()
        for col in data.df_diamonds:
            try:
                if mean(data.df_diamonds[col]) > 3:
                    cols.append(col)
            except:
                pass
        inverse_cols = [col for col in data.df_diamonds if col not in cols]
        df_if = data.df_diamonds[inverse_cols]
        self.assertTrue(
            df_if.equals(data.df_diamonds >> gr.tf_drop_if(lambda col: mean(col) > 3))
        )
        # test 2: use and
        # return colums where both conditions are false:
        # the mean greater than 3, and max < 50
        # again, this will include non-numeric columns
        cols = list()
        for col in data.df_diamonds:
            try:
                if mean(data.df_diamonds[col]) > 3 and max(data.df_diamonds[col]) < 50:
                    cols.append(col)
            except:
                pass
        inverse_cols = [col for col in data.df_diamonds if col not in cols]
        df_if = data.df_diamonds[inverse_cols]
        self.assertTrue(
            df_if.equals(
                data.df_diamonds
                >> gr.tf_drop_if(lambda col: mean(col) > 3 and max(col) < 50)
            )
        )
        # test 3: use or
        # this will return a dataframe where either of the two conditions are false:
        # the mean is greater than 3, or the max < 6
        cols = list()
        for col in data.df_diamonds:
            try:
                if mean(data.df_diamonds[col]) > 3 or max(data.df_diamonds[col]) < 6:
                    cols.append(col)
            except:
                pass
        inverse_cols = [col for col in data.df_diamonds if col not in cols]
        df_if = data.df_diamonds[inverse_cols]
        self.assertTrue(
            df_if.equals(
                data.df_diamonds
                >> gr.tf_drop_if(lambda col: mean(col) > 3 or max(col) < 6)
            )
        )
        # test 4: string operations - contain a specific string
        # this will drop any columns if they contain the word 'Ideal'
        cols = list()
        for col in data.df_diamonds:
            try:
                if any(data.df_diamonds[col].str.contains("Ideal")):
                    cols.append(col)
            except:
                pass
        inverse_cols = [col for col in data.df_diamonds if col not in cols]
        df_if = data.df_diamonds[inverse_cols]
        self.assertTrue(
            df_if.equals(
                data.df_diamonds
                >> gr.tf_drop_if(lambda col: any(col.str.contains("Ideal")))
            )
        )
        # test 5: drop any text columns
        # uses the special '.' regex symbol to find any text value
        cols = list()
        for col in data.df_diamonds:
            try:
                if any(data.df_diamonds[col].str.contains(".")):
                    cols.append(col)
            except:
                pass
        inverse_cols = [col for col in data.df_diamonds if col not in cols]
        df_if = data.df_diamonds[inverse_cols]
        self.assertTrue(
            df_if.equals(
                data.df_diamonds
                >> gr.tf_drop_if(lambda col: any(col.str.contains(".")))
            )
        )
