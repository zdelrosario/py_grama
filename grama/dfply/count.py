__all__ = [
    "tran_count",
    "tf_count",
]

from .base import dfdelegate
from .. import add_pipe
from pandas import DataFrame

@dfdelegate
def tran_count(df, *args):
    r"""Count occurrences according to groups
    """
    # Compute the count
    df_res = DataFrame(
        df.groupby([*args]).size()
    ).reset_index()
    # Change column name
    col = list(df_res.columns)
    col[-1] = "n"
    df_res.columns = col

    return df_res

tf_count = add_pipe(tran_count)
