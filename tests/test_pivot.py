import re
import unittest

from context import grama as gr
from context import data
from numpy import NaN, random
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
            names_to="var",
            values_to="val"
        )

        long_index = list(long["idx"])
        mod = max(long_index)
        single_index = list(range(0, mod))

        result = False
        if single_index == long_index[0:mod]:
            result = True

        self.assertTrue(result)


    def test_pivot_longer_select(self):
        """ Test if pivot_longer is compatible with gr.tran_select or any input
            of a DataFrame instead as 'columns' argument
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            columns=(gr.tran_select(stang,gr.matches("\\d+"))),
            names_to="var",
            values_to="val"
        )
        expected = gr.tran_pivot_longer(
            stang,
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to="var",
            values_to="val"
        )

        assert_frame_equal(long, expected)


    def test_pivot_longer_matches(self):
        """ Test if pivot_longer is compatible with gr.matches as columns input
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            columns = gr.matches("\\d+"),
            names_to="var",
            values_to="val"
        )

        expected = gr.tran_pivot_longer(
            stang,
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to="var",
            values_to="val"
        )

        assert_frame_equal(long, expected)


    def test_pivot_longer_matches_error(self):
        """ Test if selection helper retrieves no matches
        """
        with self.assertRaises(ValueError):
            stang = data.df_stang_wide
            long = gr.tran_pivot_longer(
                stang,
                columns = gr.matches("0123"),
                names_to="var",
                values_to="val"
            )


    def test_pivot_longer_names_pattern(self):
        """ Test if pivot_longer properly works with names_pattern
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            names_pattern="(E|mu)_(\\d+)",
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to=("property", "angle"),
            values_to="val"
        )

        names_to = ["property","angle"]
        names_to_check = [x for x in long.columns.values if x in names_to]

        result = False
        if names_to == names_to_check:
            result = True

        self.assertTrue(result)


    def test_pivot_longer_names_sep_and_pattern(self):
        """ Test if pivot_longer raises a ValueError if both names_sep and
            names_pattern are called
        """
        with self.assertRaises(ValueError):
            stang = data.df_stang_wide
            long = gr.tran_pivot_longer(
                stang,
                names_pattern="(E|mu)_(\\d+)",
                names_sep="_",
                columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
                names_to=("property", "angle"),
                values_to="val"
            )


    def test_pivot_longer_names_sep(self):
        """ Test if pivot_longer properly works with names_sep argument
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            names_sep="_",
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to=("property", "angle"),
            values_to="val"
        )

        names_to = ["property","angle"]
        names_to_check = [x for x in long.columns.values if x in names_to]

        result = False
        if names_to == names_to_check:
            result = True

        self.assertTrue(result)


    def test_pivot_longer_names_sep_regex(self):
        """ Test if pivot_longer properly works with names_sep argument being
            regex expression
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            names_sep=r'\_',
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to=("property", "angle"),
            values_to="val"
        )
        names_to = ["property","angle"]
        names_to_check = [x for x in long.columns.values if x in names_to]

        result = False
        if names_to == names_to_check:
            result = True

        self.assertTrue(result)


    def test_pivot_longer_names_sep_position(self):
        """ Test if pivot_longer works with names_sep argument being a position
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            names_sep=[-3],
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to=("property", "angle"),
            values_to="val"
        )
        names_to = ["property","angle"]
        names_to_check = [x for x in long.columns.values if x in names_to]

        result = False
        if names_to == names_to_check:
            result = True

        self.assertTrue(result)


    def test_pivot_longer_names_sep_thrice(self):
        """ Test if pivot_longer properly works with names_sep having to split
            columns into 3 or more
        """
        wide = gr.df_make(A_1_hello=[1,2,3], B_2_bye=[4,5,6])
        long = gr.tran_pivot_longer(
            wide,
            names_sep="_",
            columns=["A_1_hello","B_2_bye"],
            names_to=("letter","num","saying"),
            values_to="val"
        )
        names_to = ["letter","num","saying"]
        names_to_check = [x for x in long.columns.values if x in names_to]

        result = False
        if names_to == names_to_check:
            result = True

        self.assertTrue(result)


    def test_pivot_longer_names_sep_position_thrice(self):
        """ Test if pivot_longer works with names_sep argument being a position
        """
        wide = gr.df_make(A_1_hello=[1,2,3], B_2_bye=[4,5,6])
        long = gr.tran_pivot_longer(
            wide,
            names_sep=[1, 3],
            columns=["A_1_hello","B_2_bye"],
            names_to=("letter","num","saying"),
            values_to="val"
        )
        names_to = ["letter","num","saying"]
        names_to_check = [x for x in long.columns.values if x in names_to]

        result = False
        if names_to == names_to_check:
            result = True

        self.assertTrue(result)


    def test_pivot_longer_names_sep_multiple_seps(self):
        """ Test if pivot_longer properly works with names_sep having column
            names with varying amount of seps
        """
        wide = gr.df_make(A_1_hello=[1,2,3], B_2=[4,5,6])
        long = gr.tran_pivot_longer(
            wide,
            names_sep="_",
            columns=["A_1_hello","B_2"],
            names_to=("letter","num","saying"),
            values_to="val"
        )
        names_to = ["letter","num","saying"]
        names_to_check = [x for x in long.columns.values if x in names_to]

        result = False
        if names_to == names_to_check:
            result = True

        self.assertTrue(result)


    def test_pivot_longer_names_sep_and_index_to(self):
        """ Test if pivot_longer works with names_sep and index_to arguments
            together
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            index_to="idx",
            names_sep="_",
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to=("property", "angle"),
            values_to="val"
        )

        long_index = list(long["idx"])
        mod = max(long_index)
        single_index = list(range(0, mod))

        check = ["property","angle"]
        col_check = [x for x in long.columns.values if x in check]

        result = False
        if single_index == long_index[0:mod]:
            if check == col_check:
                result = True

        self.assertTrue(result)


    def test_pivot_longer_names_sep_names_to_is_1(self):
        """ Test if pivot_longer raises an error for only giving one value to
            names_to and calling names_sep
        """
        with self.assertRaises(TypeError):
            stang = data.df_stang_wide
            long = gr.tran_pivot_longer(
                stang,
                names_sep="_",
                columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
                names_to="property",
                values_to="val"
            )


    def test_pivot_longer_dot_value(self):
        """ Test pivot_longer when it receives the .value input for names_to
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to=".value",
            values_to="val"
        )
        check = ["E_00","mu_00","E_45","mu_45","E_90","mu_90"]
        col_check = [x for x in long.columns.values if x in check]

        result = False
        if set(col_check) == set(check):
            result = True

        self.assertTrue(result)

    def test_pivot_longer_dot_value_and_names_sep(self):
        """ Test pivot_longer when it receives the .value and names_sep
        """
        DF = gr.Intention()
        wide = gr.df_make(x=range(0, 6))
        wide = gr.tran_mutate(
            wide,
            y_Trend=DF.x**2,
            y_Variability=random.normal(size=6),
            y_Mixed=DF.x**2 + random.normal(size=6),
        )

        long = gr.tran_pivot_longer(
            wide,
            columns=["y_Trend", "y_Variability", "y_Mixed"],
            names_to=(".value", "type"),
            names_sep="_"
        )

        check = ["x", "type", "y"]
        col_check = [x for x in long.columns.values if x in check]

        result = False
        if set(col_check) == set(check):
            result = True

        self.assertTrue(result)


    def test_pivot_longer_dot_value_and_index_to(self):
        """ Test pivot_longer when it receives the .value input and an input
            for 'index_to'
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            index_to="idx",
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to=".value",
            values_to="val"
        )

        check = ["E_00","mu_00","E_45","mu_45","E_90","mu_90"]
        col_check = [x for x in long.columns.values if x in check]

        result = False
        if set(col_check) == set(check):
            if long.columns.values[0] == "idx":
                result = True

        self.assertTrue(result)


    def test_pivot_longer_dot_value_and_name(self):
        """ Test pivot_longer when it receives the .value and another input for
            names_to
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            names_sep="_",
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to=(".value", "angle"),
            values_to="val"
        )
        columns = list(long.columns.values)
        E_count = columns.count("E")
        mu_count = columns.count("mu")

        result = False
        if E_count > 0 and mu_count > 0:
            result = True

        self.assertTrue(result)


    def test_pivot_longer_dot_value_in_the_middle(self):
        """ Test pivot_longer when it receives the .value input in between at
            least 2 other inputs in 'names_to'
        """
        stang = data.df_stang_wide
        # rename columns for this test
        for i in stang.columns.values:
            stang.rename(columns={i: ('NA_'+i)},inplace=True)
        stang.rename(columns={'NA_thick': 'thick'},inplace=True)
        stang.rename(columns={'NA_alloy': 'alloy'},inplace=True)

        long = gr.tran_pivot_longer(
            stang,
            names_sep="_",
            columns=["NA_E_00","NA_mu_00","NA_E_45","NA_mu_45","NA_E_90","NA_mu_90"],
            names_to=("null", ".value", "angle"),
            values_to="val"
        )

        columns = list(long.columns.values)
        null_count = columns.count("null")
        E_count = columns.count("E")
        mu_count = columns.count("mu")

        result = False
        if E_count > 0 and mu_count > 0 and null_count > 0:
            result = True

        self.assertTrue(result)

        # undo column renaming for future tests
        null = 'NA_'
        for i in stang.columns.values:
            if null in i:
                stang.rename(columns={i: i.lstrip(null)},inplace=True)


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
                names_to="var",
                values_to="val"
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
            names_to="var",
            values_to="val"
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
