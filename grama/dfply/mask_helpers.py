__all__ = [
    "var_in",
    "is_nan",
    "not_nan",
]

from .base import make_symbolic
from numpy import isnan, logical_not
from pandas import Series


# ------------------------------------------------------------------------------
# Mask helpers
# ------------------------------------------------------------------------------


@make_symbolic
def var_in(series, collection):
    """
    Returns a boolean series where each entry denotes inclusion in the
    provided collection. Intended for use in mask() calls.

    Args:
        series: column to compute inclusion bools
        collection: set for inclusion calcs
    """
    bools = Series([s in collection for s in series])

    return bools


@make_symbolic
def is_nan(series, inv=False):
    """Determine if boolean

    Returns a boolean series where each entry denotes NaN or not. Intended for
    use in mask() calls.

    Args:
        series (Pandas series): column to compute NaN bools
        inv (bool): Invert logic

    """
    bools = isnan(series)

    if inv:
        return logical_not(bools)
    return bools


@make_symbolic
def not_nan(series, inv=False):
    """Determine if NOT boolean

    Returns a boolean series where each entry denotes NOT NaN or yes. Intended
    for use in mask() calls.

    Args:
        series (Pandas series): column to compute NOT NaN bools
        inv (bool): Invert logic

    """
    bools = isnan(series)

    if inv:
        return bools
    return logical_not(bools)
