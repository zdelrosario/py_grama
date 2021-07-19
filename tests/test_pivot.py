import numpy as np
import unittest

from context import grama as gr
from context import data
from pandas import DataFrame
from pandas.testing import assert_frame_equal


# python -m unittest test_pivot.TestPivotWider.test_pivot_wider && python -m unittest test_pivot.TestPivotLonger.test_pivot_longer


class TestPivotLonger(unittest.TestCase):
    """Test implementation of pivot_longer
    """
    def test_pivot_longer(self):
        data = {
            "One": {"A": 1.0, "B": 2.0, "C": 3.0},
            "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
        }

        wide = DataFrame(data)
        long = gr.tran_pivot_longer(wide, columns=("One","Two"), index_to="index", names_to="columns", values_to="values")

        expected = DataFrame(
            {
                "index": ["A", "B", "C", "A", "B", "C"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )

        # print(wide)
        # print(long)
        # print(expected)
        assert_frame_equal(long, expected)

    def test_pivot_longer_bigDF(self):
        trajectory_windowed = data.df_trajectory_windowed
        long = DataFrame(trajectory_windowed)
        print(long)
        # wide = gr.tran_pivot_wider(long,indexes="t",names_from = "x",values_from="y")
        # print(wide)
        # to note all this data is very tidy and hard to pivot_wider
        # since it possess multi dependecny or lack of index column


class TestPivotWider(unittest.TestCase):
    """Test implementation of pivot_wider
    """
    def test_pivot_wider(self):
        data = {
            "index": ["A", "B", "C", "C", "B", "A"],
            "columns": ["One", "One", "One", "Two", "Two", "Two"],
            "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }

        long = DataFrame(data)
        wide = gr.tran_pivot_wider(long,indexes="index",names_from = "columns",values_from="values")

        expected = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )

        # print(long)
        # print(wide)
        # print(expected)
        assert_frame_equal(wide, expected)
