__all__ = [
    "tran_select",
    "tf_select",
    "tran_select_if",
    "tf_select_if",
    "tran_drop",
    "tf_drop",
    "tran_drop_if",
    "tf_drop_if",
    "is_numeric",
    "starts_with",
    "ends_with",
    "contains",
    "matches",
    "everything",
    "num_range",
    "one_of",
    "columns_between",
    "columns_from",
    "columns_to",
    "resolve_selection",
]

import re
from .base import (
    Intention,
    dfdelegate,
    symbolic_evaluation,
    group_delegation,
    flatten,
    make_symbolic,
)
from .. import add_pipe
from numpy import zeros, where, ones
from numpy import max as npmax
from pandas import Index, Series
from pandas.api.types import is_numeric_dtype


# ------------------------------------------------------------------------------
# Select and drop operators
# ------------------------------------------------------------------------------


def selection_context(arg, context):
    if isinstance(arg, Intention):
        arg = arg.evaluate(context)
        if isinstance(arg, Index):
            arg = list(arg)
        if isinstance(arg, Series):
            arg = arg.name
    return arg


def selection_filter(f):
    def wrapper(*args, **kwargs):
        return Intention(
            lambda x: f(
                list(x.columns),
                *(selection_context(a, x) for a in args),
                **{k: selection_context(v, x) for k, v in kwargs.items()}
            )
        )

    return wrapper


def resolve_selection(df, *args, drop=False):
    if len(args) > 0:
        args = [a for a in flatten(args)]
        ordering = []
        column_indices = zeros(df.shape[1])
        for selector in args:
            visible = where(selector != 0)[0]
            if not drop:
                column_indices[visible] = selector[visible]
            else:
                column_indices[visible] = selector[visible] * -1
            for selection in where(selector == 1)[0]:
                if not df.columns[selection] in ordering:
                    ordering.append(df.columns[selection])
    else:
        ordering = list(df.columns)
        column_indices = ones(df.shape[1])
    return ordering, column_indices


@group_delegation
@symbolic_evaluation(eval_as_selector=True)
def tran_select(df, *args):
    r"""Select columns in a DataFrame

    Down-select or re-arrange columns in a DataFrame, usually for readability. Provide specific column names, or make use of the following selection helpers:

        starts_with() - column name begins with string
        ends_with() - column name ends with string
        contains() - column name contains string
        matches() - column name matches pattern (regular expression)
        everything() - all columns not already selected; useful for re-arranging columns without dropping

    Arguments:
        df (pandas.DataFrame): DataFrame to modify
        *args (str or selection helper): Specific column name OR selection helper.

    Returns:
        DataFrame: Data with selected columns

    Examples:

        ## Setup
        import grama as gr
        DF = gr.Intention()
        ## Load example dataset
        from grama.data import df_stang_wide

        ## Move "alloy" column to left
        (
            df_stang_wide
            >> gr.tf_select("alloy", gr.everything())
        )

        ## Find columns that start with "mu_"
        (
            df_stang_wide
            >> gr.tf_select(gr.starts_with("mu"))
        )

        ## Find columns with digits in names
        (
            df_stang_wide
            >> gr.tf_select(gr.matches("\\d+"))
        )

    """
    ordering, column_indices = resolve_selection(df, *args)
    if (column_indices == 0).all():
        return df[[]]
    selection = where(
        (column_indices == npmax(column_indices)) & (column_indices >= 0)
    )[0]
    df = df.iloc[:, selection]
    if all([col in ordering for col in df.columns]):
        ordering = [c for c in ordering if c in df.columns]
        return df[ordering]
    return df


tf_select = add_pipe(tran_select)


@group_delegation
@symbolic_evaluation(eval_as_selector=True)
def tran_drop(df, *args):
    _, column_indices = resolve_selection(df, *args, drop=True)
    if (column_indices == 0).all():
        return df[[]]
    selection = where(
        (column_indices == npmax(column_indices)) & (column_indices >= 0)
    )[0]
    return df.iloc[:, selection]


tf_drop = add_pipe(tran_drop)


@dfdelegate
def tran_select_if(df, fun):
    """Selects columns where fun(ction) is true
    Args:
        fun: a function that will be applied to columns
    """

    def _filter_f(col):
        try:
            return fun(df[col])
        except:
            return False

    cols = list(filter(_filter_f, df.columns))
    return df[cols]


tf_select_if = add_pipe(tran_select_if)


@dfdelegate
def tran_drop_if(df, fun):
    """Drops columns where fun(ction) is true
    Args:
        fun: a function that will be applied to columns
    """

    def _filter_f(col):
        try:
            return fun(df[col])
        except:
            return False

    cols = list(filter(_filter_f, df.columns))
    return df.drop(cols, axis=1)


tf_drop_if = add_pipe(tran_drop_if)


@selection_filter
def starts_with(columns, prefix):
    r"""Select columns starting with a prefix, for use in tran_select()

    Args:
        prefix (str): Prefix to detect

    """
    return [c for c in columns if c.startswith(prefix)]


@selection_filter
def ends_with(columns, suffix):
    r"""Select columns ending in a suffix, for use in tran_select()

    Args:
        suffix (str): Suffix to detect

    """
    return [c for c in columns if c.endswith(suffix)]


@selection_filter
def contains(columns, substr):
    r"""Select columns containing a substring, for use in tran_select()

    Args:
        substr (str): Substring to detect

    """
    return [c for c in columns if substr in c]


@selection_filter
def matches(columns, pattern):
    r"""Select columns matching a pattern, for use in tran_select()

    Args:
        pattern (str): String pattern to match, can be a regular expression

    """
    return [c for c in columns if re.search(pattern, c)]


@selection_filter
def everything(columns):
    "Select all columns, for use in tran_select()"
    return columns


@selection_filter
def num_range(columns, prefix, range):
    colnames = [prefix + str(i) for i in range]
    return [c for c in columns if c in colnames]


@selection_filter
def one_of(columns, specified):
    return [c for c in columns if c in specified]


@selection_filter
def columns_between(columns, start_col, end_col, inclusive=True):
    if isinstance(start_col, str):
        start_col = columns.index(start_col)
    if isinstance(end_col, str):
        end_col = columns.index(end_col)
    return columns[start_col : end_col + int(inclusive)]


@selection_filter
def columns_from(columns, start_col):
    if isinstance(start_col, str):
        start_col = columns.index(start_col)
    return columns[start_col:]


@selection_filter
def columns_to(columns, end_col, inclusive=False):
    if isinstance(end_col, str):
        end_col = columns.index(end_col)
    return columns[: end_col + int(inclusive)]


@make_symbolic
def is_numeric(series):
    """Determine if column is numeric

    Returns True if the provided column is numeric, False otherwise. Intended for calls to select_if() or mutate_if().

    Args:
        bool: Boolean corresponding to the datatype of the given column

    Examples::
        import grama as gr
        from grama.data import df_diamonds

        (
            df_diamonds
            gr.tf_select_if(gr.is_numeric)
        )

    """
    return is_numeric_dtype(series)
