__all__ = [
    "tran_summarize",
    "tf_summarize",
    "tran_summarize_each",
    "tf_summarize_each",
]

from .base import dfdelegate
from .. import add_pipe
from pandas import DataFrame, Series


@dfdelegate
def tran_summarize(df, **kwargs):
    r"""Compute specified summaries

    Compute summaries: Takes values across multiple rows and combines them into fewer rows using the functions that you specify. If the DataFrame is not grouped, then there will be just one resulting row. If the DataFrame is grouped, then there will be one row per group.

    Use tran_group_by() to group by column values before a summarize.

    Use the Intention operator (usually `DF = gr.Intention()`) as a convenient way to access columns in the DataFrame.

    Some useful summary functions are:
    - mean(x) - compute the mean, see mean_lo() and mean_up() for confidence interval bounds
    - median(x) - compute the median
    - sd(x) - compute the standard deviation
    - IQR(x) - compute the interquartile range (IQR)
    - pr(x) - compute a probability, see pr_lo() and pr_up() for confidence interval bounds
    - corr(x, y) - compute the correlation between x and y

    Args:
        df (pandas.DataFrame): Data to summarize
        **kwargs (array-like or summary expression): Summaries to assign;
            the name of the argument (left of `=`) will be the new column name,
            the value of the argument (right of `=`) defines the summary

    Returns:
        DataFrame: Summarized data

    Examples:
        ## Setup
        import grama as gr
        DF = gr.Intention()
        ## Load example dataset
        from grama.data import df_diamonds

        ## Compute some summary statistics
        (
            df_diamonds
            >> gr.tf_summarize(
                carat_mean=gr.mean(DF.carat),
                carat_median=gr.median(DF.carat),
                carat_sd=gr.sd(DF.carat),
                carat_iqr=gr.IQR(DF.carat),
            )
        )

        ## Compute statistics across cuts
        (
            df_diamonds
            >> gr.tf_group_by(DF.cut)
            >> gr.tf_summarize(
                carat_mean=gr.mean(DF.carat),
                carat_median=gr.median(DF.carat),
                carat_sd=gr.sd(DF.carat),
                carat_iqr=gr.IQR(DF.carat),
            )
        )

    """
    return DataFrame({k: [v] for k, v in kwargs.items()})

tf_summarize = add_pipe(tran_summarize)


@dfdelegate
def tran_summarize_each(df, functions, *args):
    columns, values = [], []
    for arg in args:
        if isinstance(arg, Series):
            varname = arg.name
            col = arg
        elif isinstance(arg, str):
            varname = arg
            col = df[varname]
        elif isinstance(arg, int):
            varname = df.columns[arg]
            col = df.iloc[:, arg]

        for f in functions:
            fname = f.__name__
            columns.append("_".join([varname, fname]))
            values.append(f(col))

    return DataFrame([values], columns=columns)

tf_summarize_each = add_pipe(tran_summarize_each)
