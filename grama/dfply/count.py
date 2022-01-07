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

    Count the number of rows within a specified grouping.

    Arguments
        df (DataFrame): Data to compute counts over.
        *args (str or gr.Intention()): Columns for grouping.

    Returns:
        DataFrame: Result of group counting.

    Examples:
        >>> import grama as gr
        >>> from grama.data import df_diamonds
        >>> DF = gr.Intention()
        >>> (
        >>>     # Single group variable
        >>>     df_diamonds
        >>>     >> gr.tf_count(DF.cut)
        >>> )
        >>>
        >>> (
        >>>     # Multiple group variables
        >>>     df_diamonds
        >>>     >> gr.tf_count(DF.cut, DF.clarity)
        >>> )

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
