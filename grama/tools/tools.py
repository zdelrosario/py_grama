__all__ = [
    "pipe",
    "df_equal"
]

import pandas as pd
import warnings

from functools import wraps

## Core helper functions
##################################################
## Pipe decorator
class pipe(object):
    __name__ = "pipe"

    def __init__(self, function):
        # @wraps(function) # Preserve documentation?

        self.function = function
        self.__doc__ = function.__doc__

        self.chained_pipes = []

    def __rshift__(self, other):
        assert isinstance(other, pipe)
        self.chained_pipes.append(other)
        return self

    def __rrshift__(self, other):
        other_copy = other.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(other, pd.DataFrame):
                other_copy._grouped_by = getattr(other, '_grouped_by', None)
                other_copy._plot_info = getattr(other, '_plot_info', None)
                other_copy._meta = getattr(other, '_meta', None)

        result = self.function(other_copy)

        for p in self.chained_pipes:
            result = p.__rrshift__(result)
        return result

    def __call__(self, *args, **kwargs):
        return pipe(lambda x: self.function(x, *args, **kwargs))

## DataFrame equality checker
def df_equal(df1, df2):
    """Check that two dataframes have the same columns and values. Allow
    column order to differ.

    @param df1 [DataFrame]
    @param df2 [DataFrame]

    @returns [bool]
    """

    if not set(df1.columns) == set(df2.columns):
        return False

    return df1[df2.columns].equals(df2)
