__all__ = [
    "plot_sinew_inputs",
    "pt_sinew_inputs",
    "plot_sinew_outputs",
    "pt_sinew_outputs"
]

import pandas as pd
from ..tools import pipe
from toolz import curry
import seaborn as sns
import matplotlib.pyplot as plt

## Sinew plots
@curry
def plot_sinew_inputs(df, var=None, sweep_ind="sweep_ind"):
    """
    Inspect the design
    """
    if var is None:
        raise ValueError("Must provide input columns list as keyword var")

    ## Plot
    return sns.pairplot(data=df, vars=var, hue=sweep_ind)

@pipe
def pt_sinew_inputs(*args, **kwargs):
    return plot_sinew_inputs(*args, **kwargs)

@curry
def plot_sinew_outputs(df, var=None, out=None, sweep_ind="sweep_ind", sweep_var="sweep_var"):
    """
    Construct sinew plot
    """
    if var is None:
        raise ValueError("Must provide input columns list as keyword arg var")
    if out is None:
        raise ValueError("Must provide output columns list as keyword arg out")

    ## Prepare data
    # Gather inputs
    id_vars = [col for col in df.columns if col not in var]
    df_tmp = pd.melt(
        df,
        id_vars,
        var,
        "_var",
        "_x"
    )

    # Gather outputs
    id_vars = [col for col in df_tmp.columns if col not in out]
    df_plot = pd.melt(
        df_tmp,
        id_vars,
        out,
        "_out",
        "_y"
    )

    # Filter off-sweep values
    df_plot = df_plot[df_plot[sweep_var] == df_plot["_var"]]

    # Plot
    return sns.relplot(
        data=df_plot,
        x="_x",
        y="_y",
        hue=sweep_ind,
        col="_var",
        row="_out",
        kind="line",
        facet_kws=dict(sharex=False, sharey=False)
    )


@pipe
def pt_sinew_outputs(*args, **kwargs):
    return plot_sinew_outputs(*args, **kwargs)
