__all__ = [
    "tran_pivot_longer",
    "tf_pivot_longer",
    "tran_pivot_wider",
    "tf_pivot_wider",
]


import re
from grama import add_pipe, tran_select, symbolic_evaluation, \
    group_delegation, resolve_selection, Intention
from numpy import max as npmax
from numpy import NaN, size, where, zeros
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
    names_pattern = None,
    #names_ptypes = list(),
    #names_transform = list(),
    #names_repair,
    values_to = None,
    #values_drop_na = False,
    #values_ptypes = list(),
    #values_transform = list(),
):
    """Lengthen a dataset

    "Lengthens" data by increasing the number of rows and decreasing the
    number of columns.

    Args:
        df (DataFrame): DataFrame passed through
        columns (str): Label of column(s) to pivot into longer format
        index_to(str): str name to create a new representation index of observations; Optional.
        names_to (str): name to use for the 'variable' column, if None frame.columns.name
                        is used or ‘variable’
                          • .value indicates that component of the name defines
                            the name of the column containing the cell values,
                            overriding values_to
        names_sep (str OR list of int): delimter to seperate the values of the argument(s) from
                        the 'columns' parameter into 2 new columns with those
                        values split by that delimeter
                          • Regex expression is a valid input for names_sep
        names_pattern (str): Regular expression with capture groups to define targets for names_to.
        values_to (str): name to use for the 'value' column; overridden if ".value" is provided in names_to argument.

    Notes:
        Only one of names_sep OR names_pattern may be given.

    Returns:
        DataFrame: result of being pivoted into a longer format

    Examples::

        import grama as gr
        ## Simple example
        (
            gr.df_make(
                A=[1, 2, 3],
                B=[4, 5, 6],
                C=[7, 8, 9],
            )
            >> gr.tf_pivot_longer(
                columns=["A", "B", "C"],
                names_to="variable",
                values_to="value",
            )
        )

        ## Matching columns on patterns
        (
            gr.df_make(
                x1=[1, 2, 3],
                x2=[4, 5, 6],
                x3=[7, 8, 9],
            )
            >> gr.tf_pivot_longer(
                columns=gr.matches("\\d+"),
                names_to="variable",
                values_to="value",
            )
        )

        ## Separating column names and data on a names_pattern
        (
            gr.df_make(
                E00=[1, 2, 3],
                E45=[4, 5, 6],
                E90=[7, 8, 9],
            )
            >> gr.tf_pivot_longer(
                columns=gr.matches("\\d+"),
                names_to=[".value", "angle"],
                names_pattern="(E)(\\d+)",
            )
        )

    """


    ########### Pre-Check List #############
    ### Check if tran_select was used
    if isinstance(columns, DataFrame):
        columns = columns.columns.values

    ### Check if selection helper was used:
    if isinstance(columns,Intention):
        columns = pivot_select(df, columns)
        if size(columns) == 0:
            raise ValueError("""Selection helper has found no matches. Revise
                columns input.""")

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

    if names_pattern and names_sep:
        raise ValueError("""Both names_sep and names_pattern were used,
            only one or the other is required""")


    #######################################


    ########### .value pivot #############

    ### Check if .value operation needs to occur
    if dot_value is True:

        ### collect unused columns to pivot around
        data_index = collect_indexes(df, columns)

        if names_sep is not None or names_pattern is not None:
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
                names_pattern=names_pattern,
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
    if names_sep is not None or names_pattern is not None:

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
                names_pattern=names_pattern,
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
            names_pattern=names_pattern,
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
    """Widen a dataset

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

     Examples::

        import grama as gr
        ## Simple example
        (
            gr.df_make(var=["x", "x", "y", "y"], value=[0, 1, 2, 3])
            >> gr.tf_pivot_wider(
                names_from="var",
                values_from="value",
            )
        (

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
    names_pattern,
    names_sep,
    values_to
):
    """
        split_cleanup cleans up pivots that use the names_sep functionality
    """
    ### clean up DataFrame
    # split columns based on character/regex or position via int
    if names_sep:

        # if a positional argument is used
        if isinstance(names_sep, list):
            split = longer.split.str
            names_sep.sort()

            # if one positional split is called
            if len(names_sep) == 1:
                left, right = split[:names_sep[0]], split[names_sep[0]+1:]
                split_columns = DataFrame({
                    0:left,
                    1:right
                })
            # if multiple positional splits are called
            else:
                splits = [] # split container

                # initial split
                left, right = split[:names_sep[0]], split[names_sep[0]+1:]
                splits.append(left)

                # convert and make a pointer of the remaining right half
                point = right.to_frame().split.str

                # iteratively seperate across the remaining names
                for i in range(len(names_sep)-1):
                    sep = i+1                 # current seperator
                    offset = names_sep[sep-1] # previous seperator value

                    left = point[:(names_sep[sep]-offset-1)]  # extra -1 to offset
                    right = point[names_sep[sep]-offset-1+1:] # range(len(x)-1)
                    splits.append(left)
                    point = right.to_frame().split.str # convert and re-point

                splits.append(right) # append final split

                # create final DataFrame of split values
                split_columns = DataFrame()
                for i, name in enumerate(splits):
                    column = name.to_frame()
                    split_columns[i] = column

        # if a string argument is used
        else:
            split_columns = longer.split.str.split(
                names_sep,
                expand=True,
                n=0
            )

    # split columns based on names_pattern regex capture groups
    if names_pattern:
        split = Series(longer.split)
        split_columns = split.str.extract(names_pattern)

    # add back split columns to DataFrame
    longer = concat([longer,split_columns], axis=1)

    # drop column that the split came from
    longer.drop("split", axis=1, inplace=True)

    # rename columns
    for i,v in enumerate(names_to):
        longer.rename(columns={i: v},inplace=True)

    # if any values are None make them NaN
    for i,v in enumerate(names_to):
        for j, w in enumerate(longer[names_to[i]].values):
            if w is None:
                longer[names_to[i]].values[j] = NaN

    # reorder values column to the end
    longer = longer[[c for c in longer if c not in values_to] + [values_to]]

    return longer


def collect_indexes(df, columns):
    """
        collect_indexes finds unused column to pivot around
    """
    ### look for unused columns to pivot around
    data_used = columns
    data_columns = df.columns.values
    data_index = [x for x in data_columns if x not in data_used]

    return data_index


def index_to_cleanup(df, longer, data_index):
    """
        index_to_cleanup cleans up longer if index_to was called to the function
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
            if isnull(longer[index].values[i]):
                longer[index].values[i] = longer[index].values[i%length]

    return longer


@group_delegation
@symbolic_evaluation(eval_as_selector=True)
def pivot_select(df, columns):
    """
        pivot_select helps resolve the use of a selection helper in the columns
        argument (e.g: columns = gr.matches("\\d+"))
    """
    ordering, column_indices = resolve_selection(df, columns)
    if (column_indices == 0).all():
        return df[[]]
    selection = where(
        (column_indices == npmax(column_indices)) & (column_indices >= 0)
    )[0]
    df = df.iloc[:, selection]

    return df.columns.values
