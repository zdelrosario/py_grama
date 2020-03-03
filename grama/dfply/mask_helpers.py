from .base import *
import pandas as pd

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
    bools = pd.Series([s in collection for s in series])

    return bools
