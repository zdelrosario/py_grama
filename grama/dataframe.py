__all__ = [
    "df_equal",
    "df_make",
    "df_grid",
]

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from .dfply import Intention
from .tools import tran_outer

## Safe length-checker
def safelen(x):
    try:
        if isinstance(x, str):
            raise TypeError
        return len(x)

    except TypeError:
        return 1


## DataFrame constructor utility
def df_make(**kwargs):
    r"""Construct a DataFrame

    Helper function to construct a DataFrame. A common use-case is to use df_make() to pass values to the df (and related) keyword arguments succinctly.

    Kwargs:
        varname (iterable): Column for constructed dataframe; column name inferred from variable name.

    Returns:
        DataFrame: Constructed DataFrame

    Preconditions:
        All provided iterables must have identical length or be of length one.
        All provided variable names (keyword arguments) must be distinct.

    Examples::

        import grama as gr
        from models import make_test
        md = make_test()
        (
            md
            >> gr.ev_sample(
                n=1e3,
                df_det=gr.df_make(x2=[1, 2])
            )
        )

    """
    ## Catch passed Intention operator
    if any([isinstance(v, Intention) for v in kwargs.values()]):
        raise ValueError(
            "df_make() does not support the Intention operator; " +
            "did you mean to use a DataFrame argument?\n\n" +
            "A common mistake is to write\n\n" +
            "    lambda df: gr.df_make(y=DF.x) # Incorrect\n\n" +
            "rather than\n\n" +
            "    lambda df: gr.df_make(y=df.x) # Correct"
        )

    ## Check lengths
    lengths = [safelen(v) for v in kwargs.values()]
    length_max = max(lengths)

    if not all([(l == length_max) | (l == 1) for l in lengths]):
        raise ValueError("Column lengths must be identical or one.")

    ## Construct dataframe
    df_res = DataFrame()
    for key in kwargs.keys():
        try:
            if len(kwargs[key]) > 1:
                df_res[key] = kwargs[key]
            else:
                df_res[key] = [kwargs[key][0]] * length_max
        except TypeError:
            df_res[key] = [kwargs[key]] * length_max

    return df_res


## DataFrame equality checker
def df_equal(df1, df2, close=False, precision=3):
    """Check DataFrame equality

    Check that two dataframes have the same columns and values. Allows column order to differ.

    Args:
        df1 (DataFrame): Comparison input 1
        df2 (DataFrame): Comparison input 2

    Returns:
        bool: Result of comparison

    """

    if not set(df1.columns) == set(df2.columns):
        return False

    if close:
        try:
            assert_frame_equal(
                df1[df2.columns],
                df2,
                check_dtype=False,
                check_exact=False,
                rtol=1e5
            )
            return True
        except:
            return False
    else:
        return df1[df2.columns].equals(df2)

## DataFrame constructor utility; outer product
def df_grid(**kwargs):
    r"""Construct a DataFrame as outer-product

    Helper function to construct a DataFrame as an outer-product of the given columns.

    Kwargs:
        varname (iterable): Column for constructed dataframe; column name inferred from variable name.

    Returns:
        DataFrame: Constructed DataFrame

    Preconditions:
        All provided variable names (keyword arguments) must be distinct.

    Examples::

        import grama as gr
        ## Make an empty DataFrame
        gr.df_grid()
        ## Create a row for every pair of values (6 rows total)
        gr.df_grid(x=["A", "B"], y=[1, 2, 3])

    """
    ## Catch passed Intention operator
    if any([isinstance(v, Intention) for v in kwargs.values()]):
        raise ValueError(
            "df_grid() does not support the Intention operator; " +
            "did you mean to use a DataFrame argument?"
        )

    ## Construct dataframe
    df_res = DataFrame()
    for key in kwargs.keys():
        # Switch based on argument length
        l = safelen(kwargs[key])
        if l == 1:
            df_tmp = DataFrame(columns=[key], data=[kwargs[key]])
        else:
            df_tmp = DataFrame(columns=[key], data=kwargs[key])

        df_res = tran_outer(df_res, df_outer=df_tmp)

    return df_res
