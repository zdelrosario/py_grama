__all__ = [
    "tran_inner_join",
    "tf_inner_join",
    "tran_full_join",
    "tf_full_join",
    "tran_outer_join",
    "tf_outer_join",
    "tran_left_join",
    "tf_left_join",
    "tran_right_join",
    "tf_right_join",
    "tran_semi_join",
    "tf_semi_join",
    "tran_anti_join",
    "tf_anti_join",
    "tran_bind_rows",
    "tf_bind_rows",
    "tran_bind_cols",
    "tf_bind_cols",
]

from .base import dfdelegate
from .. import add_pipe
from pandas import concat


# ------------------------------------------------------------------------------
# SQL-style joins
# ------------------------------------------------------------------------------


def get_join_parameters(join_kwargs):
    """+
    Convenience function to determine the columns to join the right and
    left DataFrames on, as well as any suffixes for the columns.
    """

    by = join_kwargs.get("by", None)
    suffixes = join_kwargs.get("suffixes", ("_x", "_y"))
    if isinstance(by, tuple):
        left_on, right_on = by
    elif isinstance(by, list):
        by = [x if isinstance(x, tuple) else (x, x) for x in by]
        left_on, right_on = (list(x) for x in zip(*by))
    else:
        left_on, right_on = by, by
    return left_on, right_on, suffixes


@dfdelegate
def tran_inner_join(df, other, **kwargs):
    """
    Joins on values present in both DataFrames.

    Args:
        df (pandas.DataFrame): Left DataFrame (passed in via pipe)
        other (pandas.DataFrame): Right DataFrame

    Kwargs:
        by (str or list): Columns to join on. If a single string, will join
            on that column. If a list of lists which contain strings or
            integers, the right/left columns to join on.
        suffixes (list): String suffixes to append to column names in left
            and right DataFrames.

    Examples::

        import grama as gr
        df_1 = gr.df_make(key=["A", "B", "C"], x=[1, 2, 3])
        df_2 = gr.df_make(key=["B", "A", "D"], y=[4, 5, 6])
        (
            df_1
            >> gr.tf_inner_join(df_2, by="key")
        )

    """

    left_on, right_on, suffixes = get_join_parameters(kwargs)
    joined = df.merge(
        other,
        how="inner",
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
    )
    return joined


tf_inner_join = add_pipe(tran_inner_join)


@dfdelegate
def tran_full_join(df, other, **kwargs):
    """
    Joins on values present in either DataFrame. (Alternate to `outer_join`)

    Args:
        df (pandas.DataFrame): Left DataFrame (passed in via pipe)
        other (pandas.DataFrame): Right DataFrame

    Kwargs:
        by (str or list): Columns to join on. If a single string, will join
            on that column. If a list of lists which contain strings or
            integers, the right/left columns to join on.
        suffixes (list): String suffixes to append to column names in left
            and right DataFrames.

    Examples::

        import grama as gr
        df_1 = gr.df_make(key=["A", "B", "C"], x=[1, 2, 3])
        df_2 = gr.df_make(key=["B", "A", "D"], y=[4, 5, 6])
        (
            df_1
            >> gr.tf_full_join(df_2, by="key")
        )

    """

    left_on, right_on, suffixes = get_join_parameters(kwargs)
    joined = df.merge(
        other,
        how="outer",
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
    )
    return joined


tf_full_join = add_pipe(tran_full_join)


@dfdelegate
def tran_outer_join(df, other, **kwargs):
    """
    Joins on values present in either DataFrame. (Alternate to `full_join`)

    Args:
        df (pandas.DataFrame): Left DataFrame (passed in via pipe)
        other (pandas.DataFrame): Right DataFrame

    Kwargs:
        by (str or list): Columns to join on. If a single string, will join
            on that column. If a list of lists which contain strings or
            integers, the right/left columns to join on.
        suffixes (list): String suffixes to append to column names in left
            and right DataFrames.

    Examples::

        import grama as gr
        df_1 = gr.df_make(key=["A", "B", "C"], x=[1, 2, 3])
        df_2 = gr.df_make(key=["B", "A", "D"], y=[4, 5, 6])
        (
            df_1
            >> gr.tf_outer_join(df_2, by="key")
        )

    """

    left_on, right_on, suffixes = get_join_parameters(kwargs)
    joined = df.merge(
        other,
        how="outer",
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
    )
    return joined


tf_outer_join = add_pipe(tran_outer_join)


@dfdelegate
def tran_left_join(df, other, **kwargs):
    """
    Joins on values present in in the left DataFrame.

    Args:
        df (pandas.DataFrame): Left DataFrame (passed in via pipe)
        other (pandas.DataFrame): Right DataFrame

    Kwargs:
        by (str or list): Columns to join on. If a single string, will join
            on that column. If a list of lists which contain strings or
            integers, the right/left columns to join on.
        suffixes (list): String suffixes to append to column names in left
            and right DataFrames.

    Examples::

        import grama as gr
        df_1 = gr.df_make(key=["A", "B", "C"], x=[1, 2, 3])
        df_2 = gr.df_make(key=["B", "A", "D"], y=[4, 5, 6])
        (
            df_1
            >> gr.tf_left_join(df_2, by="key")
        )

    """

    left_on, right_on, suffixes = get_join_parameters(kwargs)
    joined = df.merge(
        other, how="left", left_on=left_on, right_on=right_on, suffixes=suffixes
    )
    return joined


tf_left_join = add_pipe(tran_left_join)


@dfdelegate
def tran_right_join(df, other, **kwargs):
    """
    Joins on values present in in the right DataFrame.

    Args:
        df (pandas.DataFrame): Left DataFrame (passed in via pipe)
        other (pandas.DataFrame): Right DataFrame

    Kwargs:
        by (str or list): Columns to join on. If a single string, will join
            on that column. If a list of lists which contain strings or
            integers, the right/left columns to join on.
        suffixes (list): String suffixes to append to column names in left
            and right DataFrames.

    Examples::

        import grama as gr
        df_1 = gr.df_make(key=["A", "B", "C"], x=[1, 2, 3])
        df_2 = gr.df_make(key=["B", "A", "D"], y=[4, 5, 6])
        (
            df_1
            >> gr.tf_right_join(df_2, by="key")
        )

    """

    left_on, right_on, suffixes = get_join_parameters(kwargs)
    joined = df.merge(
        other,
        how="right",
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
    )
    return joined


tf_right_join = add_pipe(tran_right_join)


@dfdelegate
def tran_semi_join(df, other, **kwargs):
    """
    Returns all of the rows in the left DataFrame that have a match
    in the right DataFrame.

    Args:
        df (pandas.DataFrame): Left DataFrame (passed in via pipe)
        other (pandas.DataFrame): Right DataFrame

    Kwargs:
        by (str or list): Columns to join on. If a single string, will join
            on that column. If a list of lists which contain strings or
            integers, the right/left columns to join on.

    Examples::

        import grama as gr
        df_1 = gr.df_make(key=["A", "B", "C"], x=[1, 2, 3])
        df_2 = gr.df_make(key=["B", "A", "D"], y=[4, 5, 6])
        (
            df_1
            >> gr.tf_semi_join(df_2, by="key")
        )

    """

    left_on, right_on, suffixes = get_join_parameters(kwargs)
    if not right_on:
        right_on = [
            col_name
            for col_name in df.columns.values.tolist()
            if col_name in other.columns.values.tolist()
        ]
        left_on = right_on
    elif not isinstance(right_on, (list, tuple)):
        right_on = [right_on]
    other_reduced = other[right_on].drop_duplicates()
    joined = df.merge(
        other_reduced,
        how="inner",
        left_on=left_on,
        right_on=right_on,
        suffixes=("", "_y"),
        indicator=True,
    ).query('_merge=="both"')[df.columns.values.tolist()]
    return joined


tf_semi_join = add_pipe(tran_semi_join)


@dfdelegate
def tran_anti_join(df, other, **kwargs):
    """
    Returns all of the rows in the left DataFrame that do not have a
    match in the right DataFrame.

    Args:
        df (pandas.DataFrame): Left DataFrame (passed in via pipe)
        other (pandas.DataFrame): Right DataFrame

    Kwargs:
        by (str or list): Columns to join on. If a single string, will join
            on that column. If a list of lists which contain strings or
            integers, the right/left columns to join on.

    Examples::

        import grama as gr
        df_1 = gr.df_make(key=["A", "B", "C"], x=[1, 2, 3])
        df_2 = gr.df_make(key=["B", "A", "D"], y=[4, 5, 6])
        (
            df_1
            >> gr.tf_anti_join(df_2, by="key")
        )

    """

    left_on, right_on, suffixes = get_join_parameters(kwargs)
    if not right_on:
        right_on = [
            col_name
            for col_name in df.columns.values.tolist()
            if col_name in other.columns.values.tolist()
        ]
        left_on = right_on
    elif not isinstance(right_on, (list, tuple)):
        right_on = [right_on]
    other_reduced = other[right_on].drop_duplicates()
    joined = df.merge(
        other_reduced,
        how="left",
        left_on=left_on,
        right_on=right_on,
        suffixes=("", "_y"),
        indicator=True,
    ).query('_merge=="left_only"')[df.columns.values.tolist()]
    return joined


tf_anti_join = add_pipe(tran_anti_join)


# ------------------------------------------------------------------------------
# Binding
# ------------------------------------------------------------------------------


@dfdelegate
def tran_bind_rows(df, other, join="outer", ignore_index=False, reset=True):
    """
    Binds DataFrames "vertically", stacking them together. This is equivalent
    to `pd.concat` with `axis=0`.

    Args:
        df (pandas.DataFrame): Top DataFrame (passed in via pipe).
        other (pandas.DataFrame): Bottom DataFrame.

    Kwargs:
        join (str): One of `"outer"` or `"inner"`. Outer join will preserve
            columns not present in both DataFrames, whereas inner joining will
            drop them.
        ignore_index (bool): Indicates whether to consider pandas indices as
            part of the concatenation (defaults to `False`).
        reset (bool): Indicates whether to reset the dataframe index after
            bindin (defaults to `True`).

    """

    df = concat(
        [df.reset_index(drop=True), other.reset_index(drop=True)],
        join=join,
        ignore_index=ignore_index,
        axis=0,
        sort=False,
    )

    if reset:
        return df.reset_index(drop=True)
    return df


tf_bind_rows = add_pipe(tran_bind_rows)


@dfdelegate
def tran_bind_cols(df, other, join="outer", ignore_index=False):
    """
    Binds DataFrames "horizontally". This is equivalent to `pd.concat` with
    `axis=1`.

    Args:
        df (pandas.DataFrame): Left DataFrame (passed in via pipe).
        other (pandas.DataFrame): Right DataFrame.

    Kwargs:
        join (str): One of `"outer"` or `"inner"`. Outer join will preserve
            rows not present in both DataFrames, whereas inner joining will
            drop them.
        ignore_index (bool): Indicates whether to consider pandas indices as
            part of the concatenation (defaults to `False`).
    """

    df = concat(
        [df.reset_index(drop=True), other.reset_index(drop=True)],
        join=join,
        ignore_index=ignore_index,
        axis=1,
        sort=False,
    )
    return df


tf_bind_cols = add_pipe(tran_bind_cols)
