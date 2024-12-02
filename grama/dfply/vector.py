__all__ = [
    "order_series_by",
    "desc",
    "coalesce",
    "case_when",
    "if_else",
    "na_if",
]

import collections
from grama import lookup
from .base import make_symbolic
from numpy import argmin, arange, unique, repeat, nan, array
from pandas import concat, DataFrame, Series, isnull, factorize


# ------------------------------------------------------------------------------
# series ordering
# ------------------------------------------------------------------------------


@make_symbolic
def order_series_by(series, order_series):
    """
    Orders one series according to another series, or a list of other
    series. If a list of other series are specified, ordering is done hierarchically
    like when a list of columns is supplied to `.sort_values()`.

    Args:
        series (:obj:`pandas.Series`): the pandas Series object to be reordered.
        order_series: either a pandas Series object or a list of pandas Series
            objects. These will be sorted using `.sort_values()` with
            `ascending=True`, and the new order will be used to reorder the
            Series supplied in the first argument.

    Returns:
        reordered `pandas.Series` object
    """

    if isinstance(order_series, (list, tuple)):
        sorter = concat(order_series, axis=1)
        sorter_columns = ["_sorter" + str(i) for i in range(len(order_series))]
        sorter.columns = sorter_columns
        sorter["series"] = series.values
        sorted_series = sorter.sort_values(sorter_columns)["series"]
        return sorted_series

    sorted_series = DataFrame(
        {"series": series.values, "order": order_series.values}
    ).sort_values("order", ascending=True)["series"]
    return sorted_series


@make_symbolic
def desc(series):
    """
    Mimics the functionality of the R desc function. Essentially inverts a
    series object to make ascending sort act like descending sort.

    Args:
        series (:obj:`pandas.Series`): pandas series to be inverted prior to
            ordering/sorting.

    Returns:
        inverted `pandas.Series`. The returned series will be numeric (integers),
            regardless of the type of the original series.

    Examples::

        First group by cut, then find the first value of price when ordering by
        price ascending, and ordering by price descending using the `desc` function.

        diamonds >> group_by(X.cut) >> summarize(carat_low=first(X.price, order_by=X.price),
                                                 carat_high=first(X.price, order_by=desc(X.price)))

                 cut  carat_high  carat_low
        0       Fair       18574        337
        1       Good       18788        327
        2      Ideal       18806        326
        3    Premium       18823        326
        4  Very Good       18818        336
    """

    return series.rank(method="min", ascending=False)


# ------------------------------------------------------------------------------
# coalesce
# ------------------------------------------------------------------------------


@make_symbolic
def coalesce(*series):
    """
    Takes the first non-NaN value in order across the specified series,
    returning a new series. Mimics the coalesce function in dplyr and SQL.

    Args:
        *series: Series objects, typically represented in their symbolic form
            (like X.series).

    Examples::

        df = pd.DataFrame({
            'a':[1,np.nan,np.nan,np.nan,np.nan],
            'b':[2,3,np.nan,np.nan,np.nan],
            'c':[np.nan,np.nan,4,5,np.nan],
            'd':[6,7,8,9,np.nan]
        })
        df >> transmute(coal=coalesce(X.a, X.b, X.c, X.d))

             coal
        0       1
        1       3
        2       4
        3       5
        4  np.nan
    """

    ### FACTORIZE SOLUTION
    # coalescer = concat(series, axis=1)
    # cols = coalescer.columns
    # series = [Series(s) for s in series]
    # min_nonna = argmin(isnull(coalescer).values, axis=1)
    # coalescer.insert(loc=0,column="nonna",value=min_nonna)
    # idx, not_cols = factorize(coalescer['nonna'])
    # print(coalescer.reindex(cols, axis=1).to_numpy()[arange(len(coalescer)), idx])

    ### BFILL SOLUTION_TEXT
    # print(array(coalescer[min_nonna].bfill(axis=1).iloc[:, 0]))

    ### ORIGINAL CODE
    # print(coalescer.lookup(arange(coalescer.shape[0]), min_nonna))

    series = [Series(s) for s in series]
    coalescer = concat(series, axis=1)
    min_nonna = argmin(isnull(coalescer).values, axis=1)
    min_nonna = [coalescer.columns[i] for i in min_nonna]

    return lookup(coalescer, arange(coalescer.shape[0]), min_nonna)


# ------------------------------------------------------------------------------
# case_when
# ------------------------------------------------------------------------------


@make_symbolic
def case_when(*conditions):
    """
    Functions as a switch statement, creating a new series out of logical
    conditions specified by 2-item lists where the left-hand item is the
    logical condition and the right-hand item is the value where that
    condition is true.

    Conditions should go from the most specific to the most general. A
    conditional that appears earlier in the series will "overwrite" one that
    appears later. Think of it like a series of if-else statements.

    The logicals and values of the condition pairs must be all the same
    length, or length 1. Logicals can be vectors of booleans or a single
    boolean (`True`, for example, can be the logical statement for the
    final conditional to catch all remaining.).

    Args:
        *conditions: Each condition should be a list with two values. The first
            value is a boolean or vector of booleans that specify indices in
            which the condition is met. The second value is a vector of values
            or single value specifying the outcome where that condition is met.

    Example::

        df = pd.DataFrame({
            'num':np.arange(16)
        })
        df >> mutate(strnum=case_when([X.num % 15 == 0, 'fizzbuzz'],
                                      [X.num % 3 == 0, 'fizz'],
                                      [X.num % 5 == 0, 'buzz'],
                                      [True, X.num.astype(str)]))

            num    strnum
        0     0  fizzbuzz
        1     1         1
        2     2         2
        3     3      fizz
        4     4         4
        5     5      buzz
        6     6      fizz
        7     7         7
        8     8         8
        9     9      fizz
        10   10      buzz
        11   11        11
        12   12      fizz
        13   13        13
        14   14        14
        15   15  fizzbuzz
    """

    lengths = []
    for logical, outcome in conditions:
        if isinstance(logical, collections.abc.Iterable):
            lengths.append(len(logical))
        if isinstance(outcome, collections.abc.Iterable) and not isinstance(
            outcome, str
        ):
            lengths.append(len(outcome))
    unique_lengths = unique(lengths)
    assert len(unique_lengths) == 1
    output_len = unique_lengths[0]

    output = []
    for logical, outcome in conditions:
        if isinstance(logical, bool):
            logical = repeat(logical, output_len)
        if isinstance(logical, Series):
            logical = logical.values
        if not isinstance(outcome, collections.abc.Iterable) or isinstance(
            outcome, str
        ):
            outcome = Series(repeat(outcome, output_len))
        outcome[~logical] = nan
        output.append(outcome)

    return coalesce(*output)


# ------------------------------------------------------------------------------
# if_else
# ------------------------------------------------------------------------------


@make_symbolic
def if_else(condition, when_true, otherwise):
    """
    Wraps creation of a series based on if-else conditional logic into a function
    call.

    Provide a boolean vector condition, value(s) when true, and value(s)
    when false, and a vector will be returned the same length as the conditional
    vector according to the logical statement.

    Args:
        condition: A boolean vector representing the condition. This is often
            a logical statement with a symbolic series.
        when_true: A vector the same length as the condition vector or a single
            value to apply when the condition is `True`.
        otherwise: A vector the same length as the condition vector or a single
            value to apply when the condition is `False`.

    Example::

        import grama as gr
        from grama.data import df_diamonds
        DF = gr.Intention()
        (
            df_diamonds
            >> gr.tf_mutate(
                # Recode nonsensical x values
                x=gr.if_else(
                    DF.x == 0
                    gr.NaN,
                    DF.x,
                )
            )
        )
    """

    if not isinstance(when_true, collections.abc.Iterable) or isinstance(
        when_true, str
    ):
        when_true = repeat(when_true, len(condition))
    if not isinstance(otherwise, collections.abc.Iterable) or isinstance(
        otherwise, str
    ):
        otherwise = repeat(otherwise, len(condition))
    assert (len(condition) == len(when_true)) and (
        len(condition) == len(otherwise)
    )

    if isinstance(when_true, Series):
        when_true = when_true.values
    if isinstance(otherwise, Series):
        otherwise = otherwise.values

    output = array(
        [when_true[i] if c else otherwise[i] for i, c in enumerate(condition)]
    )
    return output


# ------------------------------------------------------------------------------
# na_if
# ------------------------------------------------------------------------------


@make_symbolic
def na_if(series, *values):
    """
    If values in a series match a specified value, change them to `np.nan`.

    Args:
        series: Series or vector, often symbolic.
        *values: Value(s) to convert to `np.nan` in the series.
    """

    series = Series(series)
    series[series.isin(values)] = nan
    return series
