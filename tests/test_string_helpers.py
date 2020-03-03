import unittest

from context import grama as gr
from numpy import isnan

##==============================================================================
## string helper tests
##==============================================================================
class TestStringHelpers(unittest.TestCase):
    def setUp(self):
        self.s_true = "foo"
        self.s_false = "bar"
        self.s_ind1 = "afoo"

        self.S_true = ["foo"]
        self.S_false = ["bar"]
        self.S_ind1 = ["afoo"]

    def test_str_detect(self):
        self.assertTrue(gr.str_detect(self.s_true, "foo"))
        self.assertTrue(not gr.str_detect(self.s_false, "foo"))
        self.assertTrue(all(gr.str_detect(self.S_true, "foo")))
        self.assertTrue(not any(gr.str_detect(self.S_false, "foo")))

    def test_str_locate(self):
        R_ind0 = gr.str_locate(self.S_true, "foo")
        R_empty = gr.str_locate(self.S_false, "foo")

        self.assertTrue(gr.str_locate(self.s_true, "foo") == [0])
        self.assertTrue(len(gr.str_locate(self.s_false, "foo")) == 0)
        self.assertTrue(R_ind0[0] == [0])
        self.assertTrue(len(R_empty[0]) == 0)

    def test_str_which(self):
        R_ind1 = gr.str_which(self.S_ind1, "foo")
        R_nan = gr.str_which(self.S_false, "foo")

        self.assertTrue(gr.str_which(self.s_ind1, "foo") == 1)
        self.assertTrue(isnan(gr.str_which(self.s_false, "foo")))
        self.assertTrue(R_ind1[0] == 1)
        self.assertTrue(isnan(R_nan[0]))

    def test_str_count(self):
        s_c2 = "foofoo"
        s_c0 = "bar"

        S_c2 = ["foofoo"]
        S_c0 = ["bar"]

        R_c2 = gr.str_count(S_c2, "foo")
        R_c0 = gr.str_count(S_c0, "foo")

        self.assertTrue(gr.str_count(s_c2, "foo") == 2)
        self.assertTrue(gr.str_count(s_c0, "foo") == 0)
        self.assertTrue(R_c2[0] == 2)
        self.assertTrue(R_c0[0] == 0)

    def test_str_replace(self):
        self.assertTrue(gr.str_replace("foofoo", "foo", "barbar") == "barbarfoo")
        self.assertTrue(gr.str_replace("barbar", "foo", "bar") == "barbar")

        R = gr.str_replace(["foo"], "foo", "bar")
        self.assertTrue(R[0] == "bar")
