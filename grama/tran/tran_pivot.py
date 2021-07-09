__all__ = [
    "tran_pivot_longer",
    "tran_pivot_wider",
]

from grama import add_pipe
from toolz import curry


@curry
def tran_pivot_longer (
    df,
    cols,
    names_to = "name",
    #names_prefix = None,
    #names_sep = None,
    #names_pattern = None,
    #names_ptypes = list(),
    #names_transform = list(),
    #names_repair,
    values_to = "value",
    #values_drop_na = False,
    #values_ptypes = list(),
    #values_transform = list(),
    **kwargs,
):
    r"""

    "Lengthens" data by increasing the number of rows and decreasing the
    number of columns.

    Args:
        df (pandas.DataFrame): DataFrame passed through
        cols (str): Label of column to pivot into longer format

    Kwargs:
        names_to (str): A string specifying the name of the column to create
                        from the data stored in the column names of data.
                        Can be a character vector, creating multiple columns,
                        if names_sep or names_pattern is provided. In this case,
                        there are two special values you can take advantage of:
                            -NA will discard that component of the name.
                            -.value indicates that component of the name
                            defines the name of the column containing the
                            cell values, overriding values_to.

        values_to (str): A string specifying the name of the column to create
                         from the data stored in cell values. If names_to is
                         a character containing the special .value sentinel,
                         this value will be ignored, and the name of the value
                         column will be derived from part of the existing
                         column names.
    """

    if cols

    return 0

@curry
def tran_pivot_wider (
    df,
    #id_cols = None,
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
    r"""

    "Widens" data by increasing the number of columns and decreasing the
    number of rows.

    Args:
        df (pandas.DataFrame): DataFrame passed through

    kwargs:
        names_from (str):

        values_from (str):
    """


    return 0
