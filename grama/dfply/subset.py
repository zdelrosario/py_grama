from .base import *
import warnings
import numpy as np
from .. import add_pipe


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
        indices = np.array(indices)
    if isinstance(indices, int):
        indices = np.array([indices])
    if isinstance(indices, pd.Series):
        indices = indices.values

    if indices.dtype == bool:
        return df.loc[indices, :].reset_index(drop=True)
    else:
        return df.iloc[indices, :].reset_index(drop=True)

tf_row_slice = add_pipe(tran_row_slice)


# ------------------------------------------------------------------------------
# Filtering/masking
# ------------------------------------------------------------------------------


@dfdelegate
def mask(df, *args):
    mask = pd.Series(np.ones(df.shape[0], dtype=bool))
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
    if not isinstance(col, pd.Series):
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
    if not isinstance(col, pd.Series):
        col = df.columns[-1]
    else:
        col = col._name

    return df[col]

tf_pull = add_pipe(tran_pull)


@dfdelegate
def tran_dropna(df, how="any", subset=None):
   return df.dropna(how=how, subset=subset).reset_index(drop=True)

tf_dropna = add_pipe(tran_dropna)
