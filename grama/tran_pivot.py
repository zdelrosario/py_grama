__all__ = [
    "tran_pivot_longer",
    "tf_pivot_longer",
    "tran_pivot_wider",
    "tf_pivot_wider",
]

from grama import add_pipe
from pandas import DataFrame, MultiIndex, Series, pivot


    # TODO:
    # -Fix Documentation on functions/make them more clear
    # -double check on bigger datasets if they both work
    # -naming conventions?
    # -look into what other tests pandas uses and which we want
    # -add notifiers to the tests
    # -add tests for piped versions

def tran_pivot_longer (
    df,
    columns,
    index_to = None,
    names_to = None,
    #names_prefix = None,
    #names_sep = None,
    #names_pattern = None,
    #names_ptypes = list(),
    #names_transform = list(),
    #names_repair,
    values_to = None,
    #values_drop_na = False,
    #values_ptypes = list(),
    #values_transform = list(),
):
    """

    "Lengthens" data by increasing the number of rows and decreasing the
    number of columns.

    Args:
        df (pandas.DataFrame): DataFrame passed through
        columns (str, tuple, list, ndarray): Label of column(s) to pivot into longer format
        indexes_to(str): str to specify the name of the index column. If name is already present index_to = None
        names_to (str): name
        values_to (str):
    """

    return df.reset_index().melt(id_vars=index_to, var_name=names_to, value_vars=columns, value_name=values_to)

tf_pivot_longer = add_pipe(tran_pivot_longer)


def tran_pivot_wider (
     df,
     #id_cols,
     indexes,
     names_from,
     #names_prefix,
     #names_sep,
     #names_glue = None,
     #names_sort = False,
     #names_glue,
     values_from,
     #values_fill = None,
     #values_fn = None
 ):
     """

     "Widens" data by increasing the number of columns and decreasing the
     number of rows.

     Args:
         df (pandas.DataFrame): DataFrame passed through
         names_from (str):
         values_from (str):
     """

     wider = pivot(df,index=indexes,columns=names_from,values=values_from)
     wider.index.name = None
     wider.columns.name = None

     return wider

tf_pivot_wider = add_pipe(tran_pivot_wider)
