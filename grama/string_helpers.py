__all__ = [
    "str_c",
    "str_count",
    "str_detect",
    "str_extract",
    "str_locate",
    "str_replace",
    "str_replace_all",
    "str_sub",
    "str_split",
    "str_which",
    "str_to_lower",
    "str_to_upper",
    "str_to_snake",
]

import re
from grama import make_symbolic, pipe, valid_dist, param_dist, dfdelegate
from numpy import NaN, isnan
from pandas import Series


# ------------------------------------------------------------------------------
# String helpers
# - a straight port of stringr
# ------------------------------------------------------------------------------

## Vectorized Concatenate
# --------------------------------------------------
def _vec_len(x):
    try:
        if isinstance(x, str):
            raise TypeError
        return len(x)
    except TypeError:
        return 1


def _ensure_series(x, l, length):
    if (l == 1) and (not isinstance(x, Series)):
        return Series([x] * length).astype(str)

    return Series(x).astype(str).reset_index(drop=True)


@make_symbolic
def str_c(*args, sep=""):
    """
    Concatenate strings
    """
    ## Pass-through a single arg
    if len(args) < 2:
        return Series(args[0]).astype(str)

    ## Check all lengths 1 or equal
    all_lengths = [_vec_len(a) for a in args]
    if not len(set(filter(lambda l: l > 1, all_lengths))) <= 1:
        raise ValueError("All arguments must be same length or scalar")
    length = max(all_lengths)

    ## Ensure first arg is string
    res = _ensure_series(args[0], all_lengths[0], length)

    ## Iteratively concatenate
    for i in range(1, len(args)):
        tmp = _ensure_series(args[i], all_lengths[i], length)
        res = res.str.cat(tmp, sep=sep)

    return res


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


def _safe_index(l, i=0):
    try:
        return l[i]
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
        return Series([_safe_index(inds) for inds in indices])

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
@make_symbolic
def str_sub(string, start=0, end=None):
    """Extract substrings"""
    try:
        if isinstance(string, str):
            raise TypeError
        return Series([s[start:end] for s in string])

    except TypeError:
        return string[start:end]


def _extract_or_empty(string, pattern):
    match = re.search(pattern, string)

    try:
        return match.group(0)
    except AttributeError:
        return ""


@make_symbolic
def str_extract(string, pattern):
    """Extract the first regex pattern match"""

    try:
        if isinstance(string, str):
            raise TypeError
        return Series([_extract_or_empty(s, pattern) for s in string])

    except TypeError:

        return _extract_or_empty(string, pattern)


## Mutate string
# --------------------------------------------------
@make_symbolic
def str_replace(string, pattern, replacement):
    """Replace the first matched pattern in each string."""

    try:
        if isinstance(string, str):
            raise TypeError
        return Series([re.sub(pattern, replacement, s, count=1) for s in string])

    except TypeError:

        return re.sub(pattern, replacement, string, count=1)


@make_symbolic
def str_replace_all(string, pattern, replacement):
    """Replace all occurences of pattern in each string."""

    try:
        if isinstance(string, str):
            raise TypeError
        return Series([re.sub(pattern, replacement, s) for s in string])

    except TypeError:

        return re.sub(pattern, replacement, string)


## Split
# --------------------------------------------------
@make_symbolic
def str_split(string, pattern, maxsplit=0):
    """Split string into list on pattern

    Args:
        string (str or iterable[str]): String(s) to split
        pattern (str): Regex pattern on which to split

    Kwargs:
        maxsplit (int): Maximum number of splits, or 0 for unlimited

    Returns
        str or iterable[str]: List (of lists) of strings

    """

    try:
        if isinstance(string, str):
            raise TypeError
        return Series([re.split(pattern, s, maxsplit=maxsplit) for s in string])

    except TypeError:

        return re.split(pattern, string, maxsplit=maxsplit)


## Case helpers
# --------------------------------------------------
@make_symbolic
def str_to_lower(string):
    """Make string lower-case

    Args:
        string (str or iterable[str]): String(s) to modify

    Returns
        str or iterable[str]: List (of lists) of strings

    """

    try:
        if isinstance(string, str):
            raise TypeError
        return Series([s.lower() for s in string])

    except TypeError:

        return string.lower()


@make_symbolic
def str_to_upper(string):
    """Make string upper-case

    Args:
        string (str or iterable[str]): String(s) to modify

    Returns
        str or iterable[str]: List (of lists) of strings

    """

    try:
        if isinstance(string, str):
            raise TypeError
        return Series([s.upper() for s in string])

    except TypeError:

        return string.upper()


def _snake_case(s):
    return re.sub("\s+", "_", s)


@make_symbolic
def str_to_snake(string):
    """Make string snake case

    Args:
        string (str or iterable[str]): String(s) to modify

    Returns
        str or iterable[str]: List (of lists) of strings

    """

    try:
        if isinstance(string, str):
            raise TypeError
        return Series([_snake_case(s) for s in string])

    except TypeError:

        return _snake_case(string)
