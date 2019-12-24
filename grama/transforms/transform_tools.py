__all__ = [
    "tran_outer",
    "tf_outer",
    "tran_gather",
    "tf_gather",
    "tran_spread",
    "tf_spread"
]

import numpy as np
import pandas as pd

from ..tools import pipe
from toolz import curry

## DataFrame outer product
@curry
def tran_outer(df, df_outer):
    """Outer merge

    Perform an outer-merge on two dataframes.

    Args:
        df (DataFrame): Data to merge
        df_outer (DataFrame): Data to merge; outer

    Returns:
        DataFrame: Merged data

    """
    n_rows = df.shape[0]
    list_df = []

    for ind in range(df_outer.shape[0]):
        df_rep = pd.concat([df_outer.loc[[ind]]] * n_rows, ignore_index=True)
        list_df.append(pd.concat((df, df_rep), axis=1))

    return pd.concat(list_df, ignore_index=True)

@pipe
def tf_outer(*args, **kwargs):
    return tran_outer(*args, **kwargs)

## Reshape functions
# --------------------------------------------------
@curry
def tran_gather(df, key, value, cols):
    """Makes a DataFrame longer by gathering columns.

    """
    id_vars = [col for col in df.columns if col not in cols]
    id_values = cols
    var_name = key
    value_name = value

    return pd.melt(
        df,
        id_vars,
        id_values,
        var_name=var_name,
        value_name=value_name
    )

@pipe
def tf_gather(*args, **kwargs):
    return tran_gather(*args, **kwargs)

@curry
def tran_spread(df, key, value, fill=np.nan, drop=False):
    """Makes a DataFrame wider by spreading columns.

    """
    index = [col for col in df.columns if ((col != key) and (col != value))]

    df_new = df.pivot_table(
        index=index,
        columns=key,
        values=value,
        fill_value=fill
    ).reset_index()

    ## Drop extraneous info
    df_new = df_new.rename_axis(None, axis=1)
    if drop:
        df_new.drop("index", axis=1, inplace=True)

    return df_new

@pipe
def tf_spread(*args, **kwargs):
    return tran_spread(*args, **kwargs)
