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
