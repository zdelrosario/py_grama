__all__ = [
    "str_detect",
    "str_count",
    "str_locate",
    "str_replace",
    "str_which",
]

import re
from .. import dfcurry, pipe, valid_dist, param_dist
from numpy import NaN, isnan
from pandas import Series


# ------------------------------------------------------------------------------
# String helpers
# - a straight port of stringr
# ------------------------------------------------------------------------------

## Detect matches
# --------------------------------------------------
@make_symbolic
def str_detect(string, pattern):
    """Detect patterns in strings

    Detect the presence of a pattern match in a string. Note that you can use
    regular expressions in these patterns.

    Examples:
        import grama as gr
        gr.str_detect(["foo", "bar", "foo"], "foo") # Exact match
        gr.str_detect(["A1", "Bx"], "\\d")          # Detect digits

    """
    try:
        ## Escape by raise if string is single string
        if isinstance(string, str):
            raise TypeError
        return Series([not (re.search(pattern, s) is None) for s in string])

    ## Single string
    except TypeError:
        return not (re.search(pattern, string) is None)


@make_symbolic
def str_locate(string, pattern):
    """Locate all patterns in strings

    Find the indices of *all* pattern matches. Returns an array for each string.
    Note that you can use regular expressions in these patterns.

    Examples:
        import grama as gr
        gr.str_locate(["foobar", "foo", "foofoo", "bar"], "foo") # Find all pattern matches
        gr.str_which(["foobar", "foo", "foofoo", "bar"], "foo") # Find first pattern match

    """
    try:
        if isinstance(string, str):
            raise TypeError
        return Series([[m.start(0) for m in re.finditer(pattern, s)] for s in string])

    except TypeError:
        return [m.start(0) for m in re.finditer(pattern, string)]


def _safe_index(l, ind=0):
    try:
        return l[ind]
    except IndexError:
        return NaN


@make_symbolic
def str_which(string, pattern):
    """Locate first pattern in strings.

    Find the index of the first pattern match. Returns indices or nan. Note that
    you can use regular expressions in these patterns.

    Examples:
        import grama as gr
        gr.str_locate(["foobar", "foo", "foofoo", "bar"], "foo") # Find all pattern matches
        gr.str_which(["foobar", "foo", "foofoo", "bar"], "foo") # Find first pattern match
        gr.str_count(["foobar", "foo", "foofoo", "bar"], "foo") # Count pattern matches

    """
    indices = str_locate(string, pattern)

    try:
        if isinstance(string, str):
            raise TypeError
        return Series([_safe_index(ind) for ind in indices])

    except TypeError:
        return _safe_index(indices)


@make_symbolic
def str_count(string, pattern):
    """Count pattern matches in strings.

    Count the number of matches in a string. Returns integers. Note that you can
    use regular expressions in these patterns.

    Examples:
        import grama as gr
        gr.str_locate(["foobar", "foo", "foofoo", "bar"], "foo") # Find all pattern matches
        gr.str_which(["foobar", "foo", "foofoo", "bar"], "foo") # Find first pattern match
        gr.str_count(["foobar", "foo", "foofoo", "bar"], "foo") # Count pattern matches

    """
    indices = str_locate(string, pattern)

    try:
        if isinstance(string, str):
            raise TypeError
        return Series([len(ind) for ind in indices])

    except TypeError:
        return len(indices)


## Subset strings
# --------------------------------------------------

## Mutate string
# --------------------------------------------------
def replace_or_none(string, pattern, replacement, ind):
    if not isnan(ind):
        return string[:ind] + replacement + string[ind + len(pattern) :]
    return string


@make_symbolic
def str_replace(string, pattern, replacement):
    """Replace the first matched pattern in each string."""

    indices = str_which(string, pattern)

    try:
        if isinstance(string, str):
            raise TypeError
        return Series(
            [
                replace_or_none(string[ind], pattern, replacement, indices[ind])
                for ind in range(len(indices))
            ]
        )

    except TypeError:

        return replace_or_none(string, pattern, replacement, indices)
