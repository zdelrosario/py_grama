__all__ = [
    "tran_mutate",
    "tf_mutate",
    "tran_mutate_if",
    "tf_mutate_if",
]

from .base import dfdelegate, make_symbolic, flatten
from .. import add_pipe


@dfdelegate
def tran_mutate(df, **kwargs):
    """Create new columns

    Creates new variables (columns) in the DataFrame specified by keyword argument pairs, where the key is the column name and the value is the new column value(s).

    Use the Intention operator (usually `DF = gr.Intention()`) as a convenient way to access columns in the DataFrame.

    Args:
        df (pandas.DataFrame): Data to modify

    Kwargs:
        **kwargs: Compute new column values
            the name of the argument (left of `=`) will be the new column name,
            the value of the argument (right of `=`) defines the new column's value

    Example:
        ## Setup
        import grama as gr
        DF = gr.Intention()
        ## Load example dataset
        from grama.data import df_diamonds

        ## Compute a rough estimate of volume
        (
            df_diamonds
            >> gr.tf_mutate(
                vol=DF.x * DF.y * DF.z
            )
            >> gr.tf_select("x", "y", "z", "vol")
        )

    """

    return df.assign(**kwargs)

tf_mutate = add_pipe(tran_mutate)


@dfdelegate
def tran_mutate_if(df, predicate, fun):
    """
    Modifies columns in place if the specified predicate is true.
    Args:
        df (pandas.DataFrame): data passed in through the pipe.
        predicate: a function applied to columns that returns a boolean value
        fun: a function that will be applied to columns where predicate returns True

    Example:
        diamonds >> mutate_if(lambda col: min(col) < 1 and mean(col) < 4, lambda row: 2 * row) >> head(3)
           carat      cut color clarity  depth  table  price     x     y     z
        0   0.46    Ideal     E     SI2   61.5   55.0    326  3.95  3.98  4.86
        1   0.42  Premium     E     SI1   59.8   61.0    326  3.89  3.84  4.62
        2   0.46     Good     E     VS1   56.9   65.0    327  4.05  4.07  4.62
        (columns 'carat' and 'z', both having a min < 1 and mean < 4, are doubled, while the
        other rows remain as they were)
    """
    cols = list()
    for col in df:
        try:
            if predicate(df[col]):
                cols.append(col)
        except:
            pass
    df[cols] = df[cols].apply(fun)
    return df

    # df2 = df.copy()
    # df2[cols] = df2[cols].apply(fun)
    # return df2

tf_mutate_if = add_pipe(tran_mutate_if)
