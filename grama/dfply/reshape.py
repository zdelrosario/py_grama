__all__ = [
    "tran_arrange",
    "tf_arrange",
    "tran_rename",
    "tf_rename",
    "tran_separate",
    "tf_separate",
    "tran_unite",
    "tf_unite",
    "tran_gather",
    "tf_gather",
    "tran_spread",
    "tf_spread",
    "tran_explode",
    "tf_explode",
    "convert_type",
]

import re
from .base import dfdelegate, symbolic_evaluation, flatten
from .. import add_pipe
from numpy import arange, nan
from pandas import Series, concat, melt, to_numeric, to_datetime


# ------------------------------------------------------------------------------
# Sorting
# ------------------------------------------------------------------------------


@dfdelegate
def tran_arrange(df, *args, **kwargs):
    """Calls `pandas.DataFrame.sort_values` to sort a DataFrame according to
    criteria.

    See:
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html

    For a list of specific keyword arguments for sort_values (which will be
    the same in arrange).

    Args:
        *args: Symbolic, string, integer or lists of those types indicating
            columns to sort the DataFrame by.

    Kwargs:
        **kwargs: Any keyword arguments will be passed through to the pandas
            `DataFrame.sort_values` function.
    """

    flat_args = [a for a in flatten(args)]

    series = [
        (
            df[arg]
            if isinstance(arg, str)
            else df.iloc[:, arg] if isinstance(arg, int) else Series(arg)
        )
        for arg in flat_args
    ]

    sorter = concat(series, axis=1).reset_index(drop=True)
    sorter = sorter.sort_values(sorter.columns.tolist(), **kwargs)
    return df.iloc[sorter.index, :].reset_index(drop=True)


tf_arrange = add_pipe(tran_arrange)


# ------------------------------------------------------------------------------
# Renaming
# ------------------------------------------------------------------------------


@dfdelegate
@symbolic_evaluation(eval_as_label=True)
def tran_rename(df, **kwargs):
    """Renames columns

    Renames columns, where keyword argument values are the current names of columns and keys are the new names.

    You can think of the keyword argument values as `newname="oldname"`; note that new variable names must follow Python variable naming conventions (no spaces, names can't start with numbers, etc.). See the Examples section below for an example of the renaming syntax.

    Args:
        df (pandas.DataFrame): DataFrame

    Kwargs:
        **kwargs: Renaming pair
            the name of the argument (left of `=`) will be the new column name,
            the value of the argument (right of `=`) is the old column name (as a string).

    Examples::

        ## Setup
        import grama as gr
        DF = gr.Intention()
        ## Load example dataset
        from grama.data import df_stang

        ## Rename columns
        (
            df_stang
            >> gr.tf_rename(
                ## Remember, the pattern is new="old"
                thickness="thick",
                alloy_name="alloy",
                elasticity="E",
                poissons_ratio="mu",
                measurement_angle="ang",
            )
        )

    """

    return df.rename(columns={v: k for k, v in kwargs.items()})


tf_rename = add_pipe(tran_rename)


# ------------------------------------------------------------------------------
# Elongate
# ------------------------------------------------------------------------------


@dfdelegate
@symbolic_evaluation(eval_as_label=["*"])
def tran_gather(df, key, values, *args, **kwargs):
    """
    Melts the specified columns in your DataFrame into two key:value columns.

    Args:
        key (str): Name of identifier column.
        values (str): Name of column that will contain values for the key.
        *args (str, int, symbolic): Columns to "melt" into the new key and
            value columns. If no args are specified, all columns are melted
            into they key and value columns.

    Kwargs:
        add_id (bool): Boolean value indicating whether to add a `"_ID"`
            column that will preserve information about the original rows
            (useful for being able to re-widen the data later).

    Example ::

        diamonds >> gather('variable', 'value', ['price', 'depth','x','y','z']) >> head(5)

           carat      cut color clarity  table variable  value
        0   0.23    Ideal     E     SI2   55.0    price  326.0
        1   0.21  Premium     E     SI1   61.0    price  326.0
        2   0.23     Good     E     VS1   65.0    price  327.0
        3   0.29  Premium     I     VS2   58.0    price  334.0
        4   0.31     Good     J     SI2   58.0    price  335.0
    """

    if len(args) == 0:
        args = df.columns.tolist()
    else:
        args = [a for a in flatten(args)]

    if kwargs.get("add_id", False):
        df = df.assign(_ID=arange(df.shape[0]))

    columns = df.columns.tolist()
    id_vars = [col for col in columns if col not in args]
    return melt(df, id_vars, list(args), key, values)


tf_gather = add_pipe(tran_gather)


# ------------------------------------------------------------------------------
# Widen
# ------------------------------------------------------------------------------


def convert_type(df, columns):
    """
    Helper function that attempts to convert columns into their appropriate
    data type.
    """
    # taken in part from the dplython package
    out_df = df.copy()
    for col in columns:
        column_values = Series(out_df[col].unique())
        column_values = column_values[~column_values.isnull()]
        # empty
        if len(column_values) == 0:
            continue
        # boolean
        if set(column_values.values) < {"True", "False"}:
            out_df[col] = out_df[col].map({"True": True, "False": False})
            continue
        # numeric
        if to_numeric(column_values, errors="coerce").isnull().sum() == 0:
            out_df[col] = to_numeric(out_df[col], errors="ignore")
            continue
        # datetime
        if to_datetime(column_values, errors="coerce").isnull().sum() == 0:
            out_df[col] = to_datetime(
                out_df[col], errors="ignore", infer_datetime_format=True
            )
            continue

    return out_df


@dfdelegate
@symbolic_evaluation(eval_as_label=["*"])
def tran_spread(df, key, values, convert=False, fill=None):
    """
    Transforms a "long" DataFrame into a "wide" format using a key and value
    column.

    If you have a mixed datatype column in your long-format DataFrame then the
    default behavior is for the spread columns to be of type `object`, or
    string. If you want to try to convert dtypes when spreading, you can set
    the convert keyword argument in spread to True.

    Args:
        key (str, int, or symbolic): Label for the key column.
        values (str, int, or symbolic): Label for the values column.

    Kwargs:
        convert (bool): Boolean indicating whether or not to try and convert
            the spread columns to more appropriate data types.


    Examples::

        widened = elongated >> spread(X.variable, X.value)
        widened >> head(5)

            _ID carat clarity color        cut depth price table     x     y     z
        0     0  0.23     SI2     E      Ideal  61.5   326    55  3.95  3.98  2.43
        1     1  0.21     SI1     E    Premium  59.8   326    61  3.89  3.84  2.31
        2    10   0.3     SI1     J       Good    64   339    55  4.25  4.28  2.73
        3   100  0.75     SI1     D  Very Good  63.2  2760    56   5.8  5.75  3.65
        4  1000  0.75     SI1     D      Ideal  62.3  2898    55  5.83   5.8  3.62
    """

    # Taken mostly from dplython package
    columns = df.columns.tolist()
    id_cols = [col for col in columns if not col in [key, values]]

    temp_index = ["" for i in range(len(df))]
    for id_col in id_cols:
        temp_index += df[id_col].map(str)

    out_df = df.assign(temp_index=temp_index)
    out_df = out_df.set_index("temp_index")
    spread_data = out_df[[key, values]]

    if not all(
        spread_data.groupby([spread_data.index, key])
        .agg("count")
        .reset_index()[values]
        < 2
    ):
        raise ValueError("Duplicate identifiers")

    spread_data = spread_data.pivot(columns=key, values=values)

    if convert and (out_df[values].dtype.kind in "OSaU"):
        columns_to_convert = [col for col in spread_data if col not in columns]
        spread_data = convert_type(spread_data, columns_to_convert)

    if not (fill is None):
        spread_data.fillna(value=fill, inplace=True)

    out_df = out_df[id_cols].drop_duplicates()
    out_df = out_df.merge(
        spread_data, left_index=True, right_index=True
    ).reset_index(drop=True)

    out_df = (out_df >> tf_arrange(id_cols)).reset_index(drop=True)

    return out_df


tf_spread = add_pipe(tran_spread)


# ------------------------------------------------------------------------------
# Separate columns
# ------------------------------------------------------------------------------


@dfdelegate
@symbolic_evaluation(eval_as_label=["*"])
def tran_separate(
    df,
    column,
    into,
    sep="[\W_]+",
    remove=True,
    convert=False,
    extra="drop",
    fill="right",
):
    """Splits one column into multiple columns.

    Split a single column containing string values into multiple columns by *separating* the strings on a specified pattern. The separator pattern is specified via the `sep` argument, and may be a regular expression.

    This verb is often used in conjunction with pivoting (e.g. tran_pivot_longer()) to separate column names from data.

    Args:
        df (pandas.DataFrame): DataFrame passed in through the pipe.
        column (str, symbolic): Label of column to split.
        into (list): List of string names for new columns.

    Kwargs:
        sep (str or list): If a string, the regex string used to split the
            column. If a list, a list of integer positions to split strings
            on.
        remove (bool): Boolean indicating whether to remove the original column.
        convert (bool): Boolean indicating whether the new columns should be converted to the appropriate type.
        extra (str): either `'drop'`, where split pieces beyond the specified new columns are dropped, or `'merge'`, where the final split piece contains the remainder of the original column.
        fill (str): either `'right'`, where `np.nan` values are filled in the right-most columns for missing pieces, or `'left'` where `np.nan` values are filled in the left-most columns.

    Returns:
        pandas.DataFrame: Modified data

    Examples::

        import grama as gr
        DF = gr.Intention

        ## Simple example
        df = gr.df_make(x=["a_1", "b_2", "c_3"])
        (
            df
            >> gr.tf_separate(
                column=DF.x,
                into=["letter", "number"],
                sep="_",
            )
        )

        ## tran_separate is helpful when pivoting data
        from grama.data import df_stang_wide
        (
            df_stang_wide
            >> gr.tf_pivot_longer(
                columns=["E_00", "mu_00", "E_45", "mu_45", "E_90", "mu_90"],
                names_to="name",
                values_to="value",
            )
            >> gr.tf_separate(
                column=DF.name,
                into=["variable", "angle"],
                sep="_",
            )
        (
    """

    assert isinstance(into, (tuple, list))

    if isinstance(sep, (tuple, list)):
        inds = [0] + list(sep)
        if len(inds) > len(into):
            if extra == "drop":
                inds = inds[: len(into) + 1]
            elif extra == "merge":
                inds = inds[: len(into)] + [None]
        else:
            inds = inds + [None]

        splits = df[column].map(
            lambda x: [
                (
                    str(x)[slice(inds[i], inds[i + 1])]
                    if i < len(inds) - 1
                    else nan
                )
                for i in range(len(into))
            ]
        )

    else:
        maxsplit = len(into) - 1 if extra == "merge" else 0
        splits = df[column].map(lambda x: re.split(sep, x, maxsplit))

    right_filler = lambda x: x + [nan for i in range(len(into) - len(x))]
    left_filler = lambda x: [nan for i in range(len(into) - len(x))] + x

    if fill == "right":
        splits = [right_filler(x) for x in splits]
    elif fill == "left":
        splits = [left_filler(x) for x in splits]

    for i, split_col in enumerate(into):
        df[split_col] = [x[i] if not x[i] == "" else nan for x in splits]

    if convert:
        df = convert_type(df, into)

    if remove:
        df.drop(column, axis=1, inplace=True)

    return df


tf_separate = add_pipe(tran_separate)


# ------------------------------------------------------------------------------
# Unite columns
# ------------------------------------------------------------------------------


@dfdelegate
@symbolic_evaluation(eval_as_label=["*"])
def tran_unite(df, colname, *args, **kwargs):
    """Unite multiple columns into one

    Does the inverse of `tran_separate`; joins columns together by a specified separator.

    Any columns that are not strings will be converted to strings.

    Args:
        df (pandas.DataFrame): DataFrame passed in through the pipe.
        colname (str): the name of the new joined column.
        *args: list of columns to be joined, which can be strings, symbolic, or
            integer positions.

    Kwargs:
        sep (str): the string separator to join the columns with.
        remove (bool): Boolean indicating whether or not to remove the
            original columns.
        na_action (str): can be one of `'maintain'` (the default),
            '`ignore'`, or `'as_string'`. The default will make the new column
            row a `NaN` value if any of the original column cells at that
            row contained `NaN`. '`ignore'` will treat any `NaN` value as an
            empty string during joining. `'as_string'` will convert any `NaN`
            value to the string `'nan'` prior to joining.

    """

    to_unite = list([a for a in flatten(args)])
    sep = kwargs.get("sep", "_")
    remove = kwargs.get("remove", True)
    # possible na_action values
    # ignore: empty string
    # maintain: keep as np.nan (default)
    # as_string: becomes string 'nan'
    na_action = kwargs.get("na_action", "maintain")

    # print(to_unite, sep, remove, na_action)

    if na_action == "maintain":
        df[colname] = df[to_unite].apply(
            lambda x: nan if any(x.isnull()) else sep.join(x.map(str)), axis=1
        )
    elif na_action == "ignore":
        df[colname] = df[to_unite].apply(
            lambda x: sep.join(x[~x.isnull()].map(str)), axis=1
        )
    elif na_action == "as_string":
        df[colname] = (
            df[to_unite].astype(str).apply(lambda x: sep.join(x), axis=1)
        )

    if remove:
        df.drop(to_unite, axis=1, inplace=True)

    return df


tf_unite = add_pipe(tran_unite)


# ------------------------------------------------------------------------------
# Nesting
# ------------------------------------------------------------------------------


@dfdelegate
@symbolic_evaluation(eval_as_label=["*"])
def tran_explode(df, col, convert=False):
    """Lengthen DataFrame by exploding iterable entries in a column.

    If you have a mixed datatype column in your long-format DataFrame then the
    default behavior is for the spread columns to be of type `object`, or
    string. If you want to try to convert dtypes when spreading, you can set
    the convert keyword argument in spread to True.

    Note: Implemented in terms of the Pandas pd.DataFrame.explode() function.

    Args:
        col (str, int, or symbolic): Label for the column to explode.

    Kwargs:
        convert (bool): Boolean indicating whether or not to try and convert
            the spread columns to more appropriate data types.

    Returns:
        DataFrame:

    Example::

    """

    df_res = df.explode(col).reset_index(drop=True)

    if convert and (df_res[col].dtype.kind in "OSaU"):
        return convert_type(df_res, [col])
    return df_res


tf_explode = add_pipe(tran_explode)
