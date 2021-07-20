import numpy as np
import unittest

from context import grama as gr
from context import data
from pandas import DataFrame, RangeIndex
from pandas.testing import assert_frame_equal


class TestPivotLonger(unittest.TestCase):
    """Test implementation of pivot_longer
    """
    def test_pivot_longer(self):
        wide = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )
        long = gr.tran_pivot_longer(wide, columns=("One","Two"),
        index_to="index", names_to="columns", values_to="values")

        expected = DataFrame(
            {
                "index": ["A", "B", "C", "A", "B", "C"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )

        assert_frame_equal(long, expected)


    def test_pivot_longer_preservation(self):
        """ Test pivot_longer preserves unsued rows if index_to is None
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            index_to = None,
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to="var",values_to="val"
        )

        expected = gr.tran_pivot_longer(
            stang,
            index_to = ["thick","alloy"],
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to="var",values_to="val"
        )

        assert_frame_equal(long, expected)


    def test_pivot_longer_preservation_index_has_no_label(self):
        """ Test if pivot_longer preserves unused index if no name
            provided for it
        """
        wide = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )
        long = gr.tran_pivot_longer(wide, columns=("One","Two"),
        index_to=None, names_to="columns", values_to="values")

        expected = DataFrame(
            {
                "index": ["A", "B", "C", "A", "B", "C"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )

        assert_frame_equal(long, expected)


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


    def test_pivot_wider_preservation(self):
        """ Test if pivot_wider preserves indexed columns pivoted around when
            indexes_from is None
        """
        stang = data.df_stang_wide
        long = gr.tran_pivot_longer(
            stang,
            index_to = ["thick","alloy"],
            columns=["E_00","mu_00","E_45","mu_45","E_90","mu_90"],
            names_to="var",values_to="val"
        )
        wide = gr.tran_pivot_wider(
            long,
            indexes_from=None,
            names_from="var",
            values_from="val"
        )
        expected = gr.tran_pivot_wider(
            long,
            indexes_from=["thick","alloy"],
            names_from="var",
            values_from="val"
        )

        assert_frame_equal(wide, expected)
