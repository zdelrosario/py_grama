__all__ = [
    "tran_outer",
    "tf_outer"
]

import numpy as np
import pandas as pd

from ..tools import pipe
from toolz import curry

## DataFrame outer product
@curry
def tran_outer(df, df_outer):
    """Perform an outer-merge on two dataframes
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
