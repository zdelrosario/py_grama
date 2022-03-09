__all__ = [
    "tran_head",
    "tf_head",
    "tran_tail",
    "tf_tail",
    "tran_sample",
    "tf_sample",
    "tran_distinct",
    "tf_distinct",
    "tran_row_slice",
    "tf_row_slice",
    "tran_filter",
    "tf_filter",
    "tran_top_n",
    "tf_top_n",
    "tran_pull",
    "tf_pull",
    "tran_dropna",
    "tf_dropna",
]

import warnings
from .base import dfdelegate, symbolic_evaluation, group_delegation
from .. import add_pipe
from numpy import array, ones
from pandas import Series


# ------------------------------------------------------------------------------
# `head` and `tail`
# ------------------------------------------------------------------------------


@dfdelegate
def tran_head(df, n=5):
    r"""Return the first n rows of a DataFrame
    """
    return df.head(n)

tf_head = add_pipe(tran_head)


@dfdelegate
def tran_tail(df, n=5):
    r"""Return the last n rows of a DataFrame
    """
    return df.tail(n)

tf_tail = add_pipe(tran_tail)


# ------------------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------------------


@dfdelegate
def tran_sample(df, *args, **kwargs):
    r"""Return a random subset of a DataFrame

    Arguments:
        n (int): Number of rows to return. Cannot be used with `frac`.
        frac (float): Fraction of items to return. Cannot be used with `n`.

    Returns:
        DataFrame: A random subset of rows from the original data.

    Notes:
        Alias for pandas DataFrame.sample(). See the docs for DataFrame.sample()
    for more information

    Examples:

        ## Setup
        import grama as gr
        ## Load example dataset
        from grama.data import df_diamonds
        ## Sample based on count
        (
            df_diamonds
            >> gr.tf_sample(n=100)
        )
        ## Sample based on fraction
        (
            df_diamonds
            >> gr.tf_sample(frac=0.01)
        )

    """
    return df.sample(*args, **kwargs).reset_index(drop=True)

tf_sample = add_pipe(tran_sample)


@group_delegation
@symbolic_evaluation(eval_as_label=["*"])
def tran_distinct(df, *args, **kwargs):
    if not args:
        return df.drop_duplicates(**kwargs).reset_index(drop=True)
    return df.drop_duplicates(list(args), **kwargs).reset_index(drop=True)

tf_distinct = add_pipe(tran_distinct)


@dfdelegate
def tran_row_slice(df, indices):
    if isinstance(indices, (tuple, list)):
        indices = array(indices)
    if isinstance(indices, int):
        indices = array([indices])
    if isinstance(indices, Series):
        indices = indices.values

    if indices.dtype == bool:
        return df.loc[indices, :].reset_index(drop=True)
    return df.iloc[indices, :].reset_index(drop=True)

tf_row_slice = add_pipe(tran_row_slice)


# ------------------------------------------------------------------------------
# Filtering/masking
# ------------------------------------------------------------------------------


@dfdelegate
def mask(df, *args):
    r"""Select rows based on provided conditions

    Select for rows based on logical conditions. This can include equality between columns `DF.x == DF.y`, comparisons with a threshold `0 <= DF.z`, or other expressions that return a boolean value.

    Use the Intention operator (usually `DF = gr.Intention()`) as a convenient way to access columns in the DataFrame.

    There are a number of helper functions that make working with filters easier / more powerful. See also:

    - var_in() : Check if given value is one of a set of values
    - is_nan() : Check if given value is not a number (nan)
    - not_nan() : Check if given value is *not* not a number (nan)
    - str_detect() : Check for the presence of a pattern in a string column

    Args:
        df (pandas.DataFrame): data passed in through the pipe.
        *args: Logical conditions

    Example:
        ## Setup
        import grama as gr
        DF = gr.Intention()
        ## Load example dataset
        from grama.data import df_diamonds

        ## Apply some filters
        (
            df_diamonds
            >> gr.tf_filter(
                ## Remove invalid dimensions
                0 < DF.x,
                0 < DF.y,
                0 < DF.z,
                ## Remove missing values
                gr.not_nan(DF.carat),
            )
        )

    """
    mask = Series(ones(df.shape[0], dtype=bool))
    for arg in args:
        if arg.dtype != bool:
            raise Exception("Arguments must be boolean.")
        mask = mask & arg.reset_index(drop=True)
    return df[mask.values].reset_index(drop=True)

tran_filter = mask  # alias for mask()

tf_filter = add_pipe(tran_filter)


@dfdelegate
def tran_top_n(df, n=None, ascending=True, col=None):
    if not n:
        raise ValueError("n must be specified")
    if not isinstance(col, Series):
        col = df.columns[-1]
    else:
        col = col._name
    index = df[[col]].copy()
    index["ranks"] = index[col].rank(ascending=ascending)
    index = index[index["ranks"] >= index["ranks"].nlargest(n).min()]
    return df.reindex(index.index)

tf_top_n = add_pipe(tran_top_n)


@dfdelegate
def tran_pull(df, col=-1):
    if not isinstance(col, Series):
        col = df.columns[-1]
    else:
        col = col._name

    return df[col]

tf_pull = add_pipe(tran_pull)


@dfdelegate
def tran_dropna(df, how="any", subset=None):
   return df.dropna(how=how, subset=subset).reset_index(drop=True)

tf_dropna = add_pipe(tran_dropna)
