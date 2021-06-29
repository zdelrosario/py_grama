__all__ = [
    "tran_group_by",
    "tf_group_by",
    "tran_ungroup",
    "tf_ungroup",
]

from .base import symbolic_evaluation
from .. import add_pipe


@symbolic_evaluation(eval_as_label=True)
def tran_group_by(df, *args):
    df._grouped_by = list(args)
    return df

tf_group_by = add_pipe(tran_group_by)


def tran_ungroup(df):
    df._grouped_by = None
    return df

tf_ungroup = add_pipe(tran_ungroup)
