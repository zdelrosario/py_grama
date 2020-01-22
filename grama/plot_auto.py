__all__ = [
    "plot_auto",
    "pt_auto",
    "plot_list"
]

import grama as gr

from grama import pipe
from toolz import curry
from pandas import melt
from seaborn import pairplot, FacetGrid, relplot
from matplotlib.pyplot import hist

## Function-specific plot functions
##################################################
## Monte Carlo
# --------------------------------------------------
@curry
def plot_monte_carlo_inputs(df, var_rand=None):
    """Inspect the design
    """
    if var_rand is None:
        raise ValueError("Must provide input columns list as keyword var_rand")

    ## Plot
    return pairplot(data=df, vars=var_rand)

@pipe
def pt_monte_carlo_inputs(*args, **kwargs):
    return plot_monte_carlo_inputs(*args, **kwargs)

@curry
def plot_monte_carlo_outputs(df, out=None):
    """Construct histograms
    """
    if out is None:
        raise ValueError("Must provide input columns list as keyword out")

    ## Gather data
    df_gathered = gr.tran_gather(df, "key", "out", out)

    ## Faceted histograms
    g = FacetGrid(df_gathered, col="key", sharex=False, sharey=False)
    g.map(hist, "out")

    return g

@pipe
def pt_monte_carlo_outputs(*args, **kwargs):
    return plot_monte_carlo_outputs(*args, **kwargs)

## Sinew plots
# --------------------------------------------------
@curry
def plot_sinew_inputs(df, var=None, sweep_ind="sweep_ind"):
    """
    Inspect the design
    """
    if var is None:
        raise ValueError("Must provide input columns list as keyword var")

    ## Plot
    return pairplot(data=df, vars=var, hue=sweep_ind)

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
    df_tmp = melt(
        df,
        id_vars,
        var,
        "_var",
        "_x"
    )

    # Gather outputs
    id_vars = [col for col in df_tmp.columns if col not in out]
    df_plot = melt(
        df_tmp,
        id_vars,
        out,
        "_out",
        "_y"
    )

    # Filter off-sweep values
    df_plot = df_plot[df_plot[sweep_var] == df_plot["_var"]]

    # Plot
    return relplot(
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

## Autoplot dispatcher
## ##################################################
plot_list = {
    "sinew_inputs": plot_sinew_inputs,
    "sinew_outputs": plot_sinew_outputs,
    "monte_carlo_inputs": plot_monte_carlo_inputs,
    "monte_carlo_outputs": plot_monte_carlo_outputs
}

@curry
def plot_auto(df):
    """Automagic plotting

    Convenience tool for various grama outputs.

    Args:
        df (DataFrame): Data output from appropriate grama routine. See
            gr.plot_list.keys() for list of supported methods.

    Returns:
        Plot results

    """
    try:
        d = df._plot_info
    except AttributeError:
        raise AttributeError("'{}' object has no attribute _plot_info. Use plot_auto() for grama outputs only.".format('DataFrame'))

    try:
        plot_fcn = plot_list[d["type"]]
    except KeyError:
        raise KeyError("'{}' Plot type not implemented.".format(d["type"]))
    plt_kwargs = {key: value for key, value in d.items() if key != "type"}

    return plot_fcn(df, **plt_kwargs)

@pipe
def pt_auto(*args, **kwargs):
    return plot_auto(*args, **kwargs)
