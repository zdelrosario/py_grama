__all__ = [
    "str_detect",
    "str_count",
    "str_locate",
    "str_replace",
    "str_which",
]

import re

from grama import dfcurry, pipe, valid_dist, param_dist
from pandas import Series
from numpy import NaN, isnan

# ------------------------------------------------------------------------------
# String helpers
# - a straight port of stringr
# ------------------------------------------------------------------------------

## Detect matches
# --------------------------------------------------
@make_symbolic
def str_detect(string, pattern):
    """
    Detect the presence of a pattern match in a string.
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
    """
    Find the indices of all pattern matches.
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
    """
    Find the index of the first pattern match.
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
    """
    Count the number of matches in a string.
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
    else:
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
