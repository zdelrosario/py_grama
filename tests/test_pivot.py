import numpy as np
import unittest

from context import core
from context import grama as gr
from context import tran

from pandas import DataFrame
from pandas.testin import assert, assert_frame_equal


class TestPivot_longer(unittest.TestCase):
    """Test implementation of pivot_longer
    """
    def test_pivot_longer(self):
        data = {
            "index": ["A", "B", "C", "C", "B", "A"],
            "columns": ["One", "One", "One", "Two", "Two", "Two"],
            "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }

        wide = DataFrame(data)
        long = tran_pivot_longer(wide, "columns")

        expected = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )

        expected.index.name, expected.columns.name = "index", "columns"
        assert_frame_equal(long, expected)

        # name tracking
        assert pivoted.index.name == "index"
        assert pivoted.columns.name == "columns"

        # don't specify values
        #pivoted = frame.pivot(index="index", columns="columns")
        #assert pivoted.index.name == "index"
        #assert pivoted.columns.names == (None, "columns")


class TestPivot_wider(unittest.TestCase):
    """Test implementation of pivot_wider
    """
    def test_pivot_wider(self):
        data = {
            "One": {"A": 1.0, "B": 2.0, "C": 3.0},
            "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
        }

        wide = DataFrame(data)
        long = tran_pivot_wider(data)

        expected = DataFrame(
            {
                "index": ["A", "B", "C", "C", "B", "A"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            }
        )

        expected.index.name, expected.columns.name = "index", "columns"
        assert_frame_equal(long, expected)

        # name tracking
        assert pivoted.index.name == "index"
        assert pivoted.columns.name == "columns"

        # don't specify values
        #pivoted = frame.pivot(index="index", columns="columns")
        #assert pivoted.index.name == "index"
        #assert pivoted.columns.names == (None, "columns")
