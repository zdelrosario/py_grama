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
    return df.head(n)

tf_head = add_pipe(tran_head)


@dfdelegate
def tran_tail(df, n=5):
    return df.tail(n)

tf_tail = add_pipe(tran_tail)


# ------------------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------------------


@dfdelegate
def tran_sample(df, *args, **kwargs):
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
