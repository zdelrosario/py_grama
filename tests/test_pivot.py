import unittest

from context import grama as gr
from context import data
from numpy import NaN
from pandas import DataFrame, RangeIndex
from pandas.testing import assert_frame_equal


class TestPivotLonger(unittest.TestCase):
    """Test implementation of pivot_longer
    """
    def test_pivot_longer(self):
        """ Test basic functionality of pivot_longer
        """
        wide = gr.df_make(One=[1,2,3], Two=[4,5,6])
        long = gr.tran_pivot_longer(
            wide,
            columns=("One","Two"),
            names_to="columns",
            values_to="values"
        )
        expected = gr.df_make(
            columns=["One","One","One","Two","Two","Two"],
            values=[1,2,3,4,5,6]
        )

        assert_frame_equal(long, expected)


    def test_pivot_longer_index_not_rangeindex(self):
        """ Test if pivot_longer makes a RangeIndex if current index is not
            a RangeIndex and preserves the orignal as new column named "index"
        """
        wide = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )
        long = gr.tran_pivot_longer(
            wide,
            columns=("One","Two"),
            names_to="columns",
            values_to="values"
        )
        expected = DataFrame(
            {
                "index": ["A", "B", "C", "A", "B", "C"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )

        assert_frame_equal(long, expected)


    def test_pivot_longer_rename_not_rangeindex(self):
        """ Test if pivot_longer makes a RangeIndex if current index is not
            a RangeIndex and preserves the orignal as a new column "index_to"
        """
        wide = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )
        long = gr.tran_pivot_longer(
            wide,
            columns=("One","Two"),
            index_to="idx",
            names_to="columns",
            values_to="values"
        )
        expected = DataFrame(
            {
                "idx": ["A", "B", "C", "A", "B", "C"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )

        assert_frame_equal(long, expected)


    def test_pivot_longer_representation_index(self):
        """ Test if pivot_longer produces a representation index for nx2
            DataFrame that has an index_to call
        """
        wide = gr.df_make(A=[1,2,3], B=[4,5,6])
        long = gr.tran_pivot_longer(
            wide,
            columns=("A", "B"),
            index_to="index",
            names_to="var",
            values_to="value"
        )
        expected = DataFrame(
            {
                "index": [0,1,2,0,1,2],
                "var": ["A","A","A","B","B","B"],
                "value": [1,2,3,4,5,6]
            }
        )

        assert_frame_equal(long, expected)


    def test_pivot_longer_no_representation_index(self):
        """ Test if pivot_longer does not produce a representation index for nx2
            DataFrame that has no index_to call
        """
        wide = gr.df_make(A=[1,2,3], B=[4,5,6])
        long = gr.tran_pivot_longer(
            wide,
            columns=("A", "B"),
            names_to="var",
            values_to="value"
        )
        expected = DataFrame(
            {
                "var": ["A","A","A","B","B","B"],
                "value": [1,2,3,4,5,6]
            }
        )

        assert_frame_equal(long, expected)


    def test_pivot_longer_rep_index_groups(self):
        """ Test if pivot_longer adds a representation index correctly when
            there is multiple groupings of observations
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            index_to = "idx",
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to="var",values_to="val"
        )
        long_index = long["idx"]
        single_index = list(range(0, max(long_index)))

        if len(long_index) == len(set(long_index)):
            if single_index == long_index:
                result = True
        else:
            result = True

        self.assertTrue(result)


class TestPivotWider(unittest.TestCase):
    """Test implementation of pivot_wider
    """
    def test_pivot_wider(self):
        long = DataFrame(
            {
                "loo": ["A", "B", "C", "A", "B", "C"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )
        wide = gr.tran_pivot_wider(
            long,
            indexes_from="loo",
            names_from="columns",
            values_from="values"
        )
        expected = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )
        expected.insert(0,"loo",["A","B","C"])
        expected.index=RangeIndex(start=0,stop=len(expected),step=1)

        assert_frame_equal(wide, expected)


    def test_pivot_wider_duplicate_entries(self):
        """ Test if pivot_wider raise an error for duplicate entries
        """
        with self.assertRaises(ValueError):
            stang = data.df_stang_wide
            long = gr.tran_pivot_longer(
                stang,
                columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
                names_to="var",values_to="val"
            )
            wide = gr.tran_pivot_wider(
                long,
                names_from="var",
                values_from="val"
            )


    def test_pivot_wider_representation_index(self):
        """ Test if pivot_wider can handle duplicate entries if a representation
            index is present
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            index_to = "idx",
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to="var",values_to="val"
        )
        wide = gr.tran_pivot_wider(
            long,
            names_from="var",
            values_from="val"
        )
        expected = gr.df_make(
            idx=[0,1,2,3,4,5,6,7,8],
            thick=[0.022,0.022,0.032,0.032,0.064,0.064,0.081,0.081,0.081],
            alloy=[" al_24st"," al_24st"," al_24st"," al_24st"," al_24st",
                " al_24st"," al_24st"," al_24st"," al_24st"],
            E_00=[10600.000,10600.000,10400.000,10300.000,10500.000,10700.000,
                10000.000,10100.000,10000.000],
            E_45=[10700.000,10500.000,10400.000,10500.000,10400.000,10500.00,
                10000.000,9900.000,-1.0],
            E_90=[10500.000,10700.000,10300.000,10400.000,10400.000,10500.000,
                9900.000,10000.000,9900.000],
            mu_00=[0.321,0.323,0.329,0.319,0.323,0.328,0.315,0.312,0.311],
            mu_45=[0.329,0.331,0.318,0.326,0.331,0.328,0.320,0.312,-1.000],
            mu_90=[0.310,0.323,0.322,0.330,0.327,0.320,0.314,0.316,0.314]
        )

        assert_frame_equal(wide, expected)


    def test_pivot_wider_representation_index_multiple_cols(self):
        long = DataFrame(
            {
                "loo": ["A", "B", "C", "A", "B", "C"],
                "foo": ["X", "Y", "Z", "X", "Y", "Z"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )
        wide = gr.tran_pivot_wider(
            long,
            indexes_from=("loo","foo"),
            names_from="columns",
            values_from="values"
        )
        expected = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )
        expected.insert(0,"loo",["A","B","C"])
        expected.insert(1,"foo",["X","Y","Z"])
        expected.index=RangeIndex(start=0,stop=len(expected),step=1)

        assert_frame_equal(wide, expected)


    def test_pivot_wider_NaN_entries(self):
        """ Test if pivot_wider returns a table with NaN values for unspecified
            entries that have no represenational index
        """
        original = gr.df_make(A=[1,2,3], B=[4,5,6])
        long = gr.tran_pivot_longer(
            original,
            columns=("A", "B"),
            names_to="var",
            values_to="value"
        )
        wide = gr.tran_pivot_wider(
            long,
            names_from="var",
            values_from="value"
        )
        expected = gr.df_make(
            A=[1,2,3,NaN,NaN,NaN],
            B=[NaN,NaN,NaN,4,5,6]
        )

        assert_frame_equal(wide, expected)
