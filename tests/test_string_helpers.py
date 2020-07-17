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

        R = gr.str_replace(["foo", "nope"], "foo", "bar")
        self.assertTrue(R[0] == "bar")
        self.assertTrue(R[1] == "nope")

    def test_str_replace_all(self):
        self.assertTrue(gr.str_replace_all("foofoo", "foo", "bar") == "barbar")

        R = gr.str_replace_all(["foofoo", "nope"], "foo", "bar")
        self.assertTrue(R[0] == "barbar")
        self.assertTrue(R[1] == "nope")

    def test_str_sub(self):
        s_base0 = "foofoo"
        s_sub0 = "foo"
        s_base1 = "f"
        s_sub1 = "f"

        S_base0 = ["foofoo", "bar"]
        S_sub0 = ["foo", "bar"]

        self.assertTrue(gr.str_sub(s_base0, end=3), s_sub0)
        self.assertTrue(gr.str_sub(s_base1, end=3), s_sub1)
        self.assertTrue(all(gr.str_sub(S_base0, end=3) == S_sub0))

    def test_str_extract(self):
        s_base0 = "foo_bar"
        s_ext0 = "_bar"
        s_base1 = "foo123"
        s_ext1 = "123"

        S_base0 = ["foo_bar", "foo_biddy"]
        S_ext0 = ["_bar", "_biddy"]

        self.assertTrue(gr.str_extract(s_base0, "_\\w+"), s_ext0)
        self.assertTrue(gr.str_extract(s_base1, "\\d+"), s_ext1)

        res = gr.str_extract(S_base0, "_\\w+")
        print(res)  ## DEBUG
        self.assertTrue(all(res == S_ext0))
