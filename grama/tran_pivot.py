__all__ = [
    "tran_pivot_longer",
    "tf_pivot_longer",
    "tran_pivot_wider",
    "tf_pivot_wider",
]


from grama import add_pipe
from pandas import DataFrame, IndexSlice, MultiIndex, RangeIndex, Series, \
    isnull, pivot, pivot_table
from pandas.api.types import is_int64_dtype


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
    values_to = None
    #values_drop_na = False,
    #values_ptypes = list(),
    #values_transform = list(),
):
    """

    "Lengthens" data by increasing the number of rows and decreasing the
    number of columns.

    Args:
        df (DataFrame): DataFrame passed through
        columns (str): Label of column(s) to pivot into longer format
        index_to(str): str name to create a new representation index of
                        observations
        names_to (str): name to use for the 'variable' column, if None frame.columns.name
                        is used or ‘variable’
        values_to (str): name to use for the 'value' column

    Returns:
        DataFrame: result of being pivoted into a longer format

    Examples:

        >>> import grama as gr
        >>> from pandas import DataFrame
        >>> wide = DataFrame(
                {
                    "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                    "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
                }
            )
        >>> long = gr.tran_pivot_longer(
                    wide,
                    columns=("One","Two"),
                    index_to="index",
                    names_to="columns",
                    values_to="values")

    """

    ### Check if index_to is provided
    if index_to is None:

        # check to see if all columns are used already
        data_columns = df.columns.values
        data_index = [x for x in data_columns if x not in columns]

        # check if data_index is empty and if it has a RangeIndex
        if not data_index:
            if is_int64_dtype(df.index.dtype):
                # if so do not add extra index column and pivot
                longer = df.reset_index().melt(
                    id_vars=None,
                    var_name=names_to,
                    value_vars=columns,
                    value_name=values_to
                )
                return(longer)

            # if it does not have a RangeIndex create new column from ID column
            # and add RangeIndex
            longer = df.reset_index().melt(
                id_vars="index",
                var_name=names_to,
                value_vars=columns,
                value_name=values_to
            )
            return(longer)

        # look for unused columns to pivot around
        data_used = columns
        data_index = [x for x in data_columns if x not in data_used]

        # pivot with leftover name that would be the index column
        if data_index:
            longer = df.reset_index().melt(
                id_vars=data_index,
                var_name=names_to,
                value_vars=columns,
                value_name=values_to
            )
            return(longer)

    ### Collect indexes to be preserved post-pivot
    data_columns = df.columns
    data_used = columns
    data_index = [x for x in data_columns if x not in data_used]

    ### Add index column to dataset
    long = df.reset_index().melt(
            id_vars="index",
            var_name=names_to,
            value_vars=columns,
            value_name=values_to
        )
    ### rename index column to desired: index_to
    long.rename(columns={'index': index_to},inplace=True)

    ### if there was columns needing to be re-inserted do so
    if data_index is not None:
        for i, v in enumerate(data_index):
            print(i)
            long.insert(
                loc=i+1,
                column=data_index[i],
                value=df[data_index[i]]
            )

    ### repair NaN values from transformation
    for index in data_index:
        length = len(df[index])
        for i, v in enumerate(long[index]):
            if isnull(long[index][i]):
                long[index][i] = long[index][i%length]

    return long

tf_pivot_longer = add_pipe(tran_pivot_longer)


def tran_pivot_wider (
     df,
     #id_cols,
     names_from,
     indexes_from = None,
     #names_prefix,
     #names_sep,
     #names_glue = None,
     #names_sort = False,
     #names_glue,
     values_from = None,
     #values_fill = None,
     #values_fn = None
 ):
    """

     "Widens" data by increasing the number of columns and decreasing the
     number of rows.

     Args:
         df (DataFrame): DataFrame passed through
         names_from (str): Column(s) name(s) to use to make new columns
         indexes_from (str): Column(s) to use to make new index, if None will
                             preserve and use unspecified columns as index
         values_from (str): Column(s) to use as new values column(s)

     Returns:
        DataFrame: result of being pivoted wider

     Example:
        >>> import grama as gr
        >>> from pandas import DataFrame
        >>> long = DataFrame(
                {
                    "index": ["A", "B", "C", "A", "B", "C"],
                    "columns": ["One", "One", "One", "Two", "Two", "Two"],
                    "values": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                }
            )
        >>> wide = gr.tran_pivot_wider(
                long,
                indexes_from="index",
                names_from="columns",
                values_from="values"
            )
    """

    if indexes_from is None:
        ### Clean columns list to find unused columns and preserve them
        data_columns = df.columns
        data_values_used = values_from
        data_columns_used = names_from
        data_1st_clean = [x for x in data_columns if x not in data_columns_used]
        data_index_clean = [x for x in data_1st_clean if x not in data_values_used]
        if not data_index_clean:
            data_index_clean = None

        ### try pivot, if no repeats exist in data_index_clean values continue,
        ### else use pivot_table to handle repreated values
        wider = pivot(
            df,
            index=data_index_clean,
            columns=names_from,
            values=values_from
        )
        ### Post cleaning of leftover column and index names
        wider.columns.name = None
        # preserve rows by re-inserting them
        if data_index_clean is not None:
            for i, v in enumerate(data_index_clean):
                wider.insert(
                    loc=i,
                    column=data_index_clean[i],
                    value=wider.index.get_level_values(data_index_clean[i])
                    )
        # remake index to be numbered column with no name
        wider.index = RangeIndex(start=0,stop=len(wider),step=1)
        wider.index.name = None

        return wider

    ### If indexes_from exists do the same repeat check as above but with the
    ### provided indexes
    wider = pivot(
        df,
        index=indexes_from,
        columns=names_from,
        values=values_from
    )
    ### Post cleaning of leftover column and index names
    wider.columns.name = None
    ## preserve rows by re-inserting them
    if isinstance(indexes_from, str):
        wider.insert(0,indexes_from,wider.index)
    # if str, insert as so, else loop through list type of indexes_from
    else:
        for i, v in enumerate(indexes_from):
            wider.insert(
                loc=i,
                column=indexes_from[i],
                value=wider.index.get_level_values(indexes_from[i])
                )
    # remake index to be numbered column with no name
    wider.index = RangeIndex(start=0,stop=len(wider),step=1)
    wider.index.name = None

    return wider

tf_pivot_wider = add_pipe(tran_pivot_wider)
