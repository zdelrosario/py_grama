__all__ = [
    "tran_pivot_longer",
    "tf_pivot_longer",
    "tran_pivot_wider",
    "tf_pivot_wider",
]


from grama import add_pipe, tran_select
from numpy import NaN
from pandas import DataFrame, IndexSlice, MultiIndex, RangeIndex, Series, \
    concat, isnull, pivot, pivot_table
from pandas.api.types import is_int64_dtype


def tran_pivot_longer (
    df,
    columns,
    index_to = None,
    names_to = None,
    #names_prefix = None,
    names_sep = None,
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
        df (DataFrame): DataFrame passed through
        columns (str): Label of column(s) to pivot into longer format
        index_to(str): str name to create a new representation index of
                        observations
        names_to (str): name to use for the 'variable' column, if None frame.columns.name
                        is used or ‘variable’
                          • .value indicates that component of the name defines
                            the name of the column containing the cell values,
                            overriding values_to
        names_sep (str): delimter to seperate the values of the argument(s) from
                        the 'columns' parameter into 2 new columns with those
                        values split by that delimeter
                          • Regex expression is a valid input for names_sep
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

    ########### Pre-Check List #############
    ### Check if tran_select was used
    if isinstance(columns, DataFrame):
        columns = columns.columns.values
    ### Check if gr.tran_select helper was used:
    # NEEDS to be implmented, currently gr.tran_select needs to provided with
    # gr.matches or other helper function for this to work

    ### Check if names_to is a list or str
    names_str = False
    if isinstance(names_to, str):
        names_str = True
        if names_sep is not None:
            raise TypeError("""In order to use names_sep more than 1 value
                needs to passed to names_to""")

    ### Check for .value input
    dot_value = False
    if names_str is False:
        for i, v in enumerate(names_to):
            if names_to[i] == ".value":
                dot_value = True
    else:
        if names_to == ".value":
            dot_value = True

    ### Check values_to argument
    if values_to is None:
        values_to = "values"

    #######################################


    ########### .value pivot #############

    ### Check if .value operation needs to occur
    if dot_value is True:

        ### collect unused columns to pivot around
        data_index = collect_indexes(df, columns)

        if names_sep is not None:
            ### Add index and split column to dataset
            longer = df.reset_index().melt(
                    id_vars="index",
                    var_name="split",
                    value_vars=columns,
                    value_name=values_to
                )

            ### DataFrame Cleanup
            longer = split_cleanup(
                longer=longer,
                names_to=names_to,
                names_sep=names_sep,
                values_to=values_to
            )

        else:
            ### Add index column and .value column
            longer = df.reset_index().melt(
                    id_vars="index",
                    var_name=".value",
                    value_vars=columns,
                    value_name=values_to
                )

        ### clean up index_to call
        longer = index_to_cleanup(df, longer, data_index)

        ### arrange what indexes_from should be
        if names_str is True:
            indexes = ["index"] + data_index
        else:
            names_to = list(names_to)
            value_loc = names_to.index(".value")
            if value_loc == 0:
                indexes = ["index"] + data_index + names_to[1:]
            else:
                indexes = ["index"] + data_index + names_to[0:value_loc] \
                    + names_to[(value_loc+1):]

        ### Pivot wider the .value column
        value_longer = tran_pivot_wider(
            longer,
            indexes_from=indexes,
            names_from=".value",
            values_from=values_to
        )

        if index_to is None:
            ### drop "index" column
            value_longer.drop("index", axis=1, inplace=True)
        else:
            ### rename index column to desired: index_to
            value_longer.rename(columns={'index': index_to},inplace=True)

        return value_longer

    #########################################


    ########### names_sep pivot #############

    ### Only if names_sep is used
    if names_sep is not None:

        ### collect unused columns to pivot around
        data_index = collect_indexes(df, columns)

        if index_to is None:
            ### initial pivoted DataFrame
            longer = df.reset_index().melt(
                id_vars=data_index,
                var_name="split",
                value_vars=columns,
                value_name=values_to
            )

            ### DataFrame Cleanup
            longer = split_cleanup(
                longer=longer,
                names_to=names_to,
                names_sep=names_sep,
                values_to=values_to
            )

            return(longer)

        ### Add index column to dataset
        longer = df.reset_index().melt(
                id_vars="index",
                var_name="split",
                value_vars=columns,
                value_name=values_to
            )
        ### rename index column to desired: index_to
        longer.rename(columns={'index': index_to},inplace=True)

        longer = index_to_cleanup(df, longer, data_index)

        ### DataFrame Cleanup
        longer = split_cleanup(
            longer=longer,
            names_to=names_to,
            names_sep=names_sep,
            values_to=values_to
        )

        return(longer)

    ######################################


    ########### normal pivot #############

    ### Check if index_to is provided
    if index_to is None:

        ### check to see if all columns are used already
        data_columns = df.columns.values
        data_index = [x for x in data_columns if x not in columns]

        ### check if data_index is empty and if it has a RangeIndex
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

        ### look for unused columns to pivot around
        data_used = columns
        data_index = [x for x in data_columns if x not in data_used]

        ### pivot with leftover name that would be the index column
        if data_index:
            longer = df.reset_index().melt(
                id_vars=data_index,
                var_name=names_to,
                value_vars=columns,
                value_name=values_to
            )
            return(longer)

    ### collect unused columns to preserve post pivot
    data_index = collect_indexes(df, columns)

    ### Add index column to dataset
    longer = df.reset_index().melt(
            id_vars="index",
            var_name=names_to,
            value_vars=columns,
            value_name=values_to
        )
    ### rename index column to desired: index_to
    longer.rename(columns={'index': index_to},inplace=True)

    longer = index_to_cleanup(df, longer, data_index)

    return longer

    ######################################

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


def split_cleanup(
    longer,
    names_to,
    names_sep,
    values_to
):
    """
        split_cleanup cleans up pivots that use the names_sep functionality
    """
    ### clean up DataFrame
    # split columns
    split_columns = longer.split.str.split(
        names_sep,
        expand=True,
        n=0
    )
    # add back split columns to DataFrame
    longer = concat([longer,split_columns], axis=1)
    # drop column that the split came from
    longer.drop("split", axis=1, inplace=True)
    # rename columns
    for i,v in enumerate(names_to):
        longer.rename(columns={i: names_to[i]},inplace=True)
    # if any values are None make them NaN
    for i,v in enumerate(names_to):
        for j, w in enumerate(longer[names_to[i]]):
            if w is None:
                longer[names_to[i]][j] = NaN
    # reorder values column to the end
    longer = longer[[c for c in longer if c not in values_to] + [values_to]]

    return(longer)


def collect_indexes(df, columns):
    """
        collect_indexes finds unused column to pivot around
    """
    ### look for unused columns to pivot around
    data_used = columns
    data_columns = df.columns.values
    data_index = [x for x in data_columns if x not in data_used]

    return(data_index)

def index_to_cleanup(df, longer, data_index):
    """
        index_to_cleanup cleansup longer if index_to was called to the function
    """
    ### if there was columns needing to be re-inserted do so
    if data_index is not None:
        for i, v in enumerate(data_index):
            longer.insert(
                loc=i+1,
                column=data_index[i],
                value=df[data_index[i]]
            )

    ### repair NaN values from transformation
    for index in data_index:
        length = len(df[index])
        for i, v in enumerate(longer[index]):
            if isnull(longer[index][i]):
                longer[index][i] = longer[index][i%length]

    return(longer)
