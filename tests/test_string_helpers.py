import unittest

from context import grama as gr
from numpy import isnan
from pandas import Series, Index

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

    def test_str_c(self):
        ## Correct errors
        with self.assertRaises(ValueError):
            gr.str_c([1, 2], [1, 2, 3])

        ## Correct behavior
        S0_true = Series(["foo"])
        S0_comp = gr.str_c("foo")
        self.assertTrue(S0_true.equals(S0_comp))

        S1_true = Series(["x0", "x1"])
        S1_comp = gr.str_c("x", [0, 1])
        self.assertTrue(S1_true.equals(S1_comp))

        S2_true = Series(["x0", "x1"])
        S2_comp = gr.str_c("x", [0, 1])
        self.assertTrue(S2_true.equals(S2_comp))

        # Catch series index issues
        S3_true = Series(["x0", "x1"])
        S_tmp = Series([0, 1])
        S_tmp.index = Index(["a", "b"])

        S3_comp = gr.str_c("x", S_tmp)
        self.assertTrue(S3_true.equals(S3_comp))

        # Catch length-one issue
        S4_true = Series(["0x"])
        S4_comp = gr.str_c(Series([0], name="test"), "x")
        self.assertTrue(S4_true.equals(S4_comp))

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

    def test_str_split(self):
        self.assertTrue(gr.str_split("x-y", "-") == ["x", "y"])

        S0 = gr.str_split(["x-y", "u-v-w"], "-")
        self.assertTrue(S0[0] == ["x", "y"])
        self.assertTrue(S0[1] == ["u", "v", "w"])

        S1 = gr.str_split(["x-y", "u-v-w"], "-", maxsplit=1)
        self.assertTrue(S1[0] == ["x", "y"])
        self.assertTrue(S1[1] == ["u", "v-w"])

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
