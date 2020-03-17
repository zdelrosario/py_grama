__all__ = [
    "plot_scattermat",
    "pt_scattermat",
    "plot_hists",
    "pt_hists",
    "plot_sinew_inputs",
    "pt_sinew_inputs",
    "plot_sinew_outputs",
    "pt_sinew_outputs",
    "plot_auto",
    "pt_auto",
    "plot_list",
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
def plot_scattermat(df, var=None):
    r"""Create a scatterplot matrix

    Create a scatterplot matrix. Often used to visualize a design (set of inputs
    points) before evaluating the functions.

    Args:
        var (list of strings): Variables to plot

    Returns:
        DataFrame: Results of evaluation or unevaluated design

    Examples:

        >>> import grama as gr
        >>> import matplotlib.pyplot as plt
        >>> from grama.models import make_cantilever_beam
        >>> md = make_cantilever_beam()
        >>> md >> \
        >>>     gr.ev_monte_carlo(n=100, df_det="nom", skip=True) >> \
        >>>     gr.pt_scattermat(var=md.var)
        >>> plt.show()

    """
    if var is None:
        raise ValueError("Must provide input columns list as keyword var")

    ## Plot
    return pairplot(data=df, vars=var)


@pipe
def pt_scattermat(*args, **kwargs):
    return plot_scattermat(*args, **kwargs)


@curry
def plot_hists(df, out=None):
    r"""Construct histograms

    Create a set of histograms. Often used to visualize the results of random
    sampling for multiple outputs.

    Args:
        out (list of strings): Variables to plot

    Returns:
        Seaborn histogram plot

    Examples:

        >>> import grama as gr
        >>> import matplotlib.pyplot as plt
        >>> from grama.models import make_cantilever_beam
        >>> md = make_cantilever_beam()
        >>> md >> \
        >>>     gr.ev_monte_carlo(n=100, df_det="nom") >> \
        >>>     gr.pt_hists(out=md.out)
        >>> plt.show()

    """
    if out is None:
        raise ValueError("Must provide input columns list as keyword out")

    ## Gather data
    df_gathered = df >> gr.tf_gather("key", "out", out)

    ## Faceted histograms
    g = FacetGrid(df_gathered, col="key", sharex=False, sharey=False)
    g.map(hist, "out").set_axis_labels("Output", "Count")

    return g


@pipe
def pt_hists(*args, **kwargs):
    return plot_hists(*args, **kwargs)


## Sinew plots
# --------------------------------------------------
@curry
def plot_sinew_inputs(df, var=None, sweep_ind="sweep_ind"):
    r"""Inspect a sinew design

    Create a scatterplot matrix with hues. Often used to visualize a sinew
    design before evaluating the model functions.

    Args:
        df (Pandas DataFrame): Input design data
        var (list of strings): Variables to plot
        sweep_ind (string): Sweep index column in df

    Returns:
        Seaborn scatterplot matrix

    Examples:

        >>> import grama as gr
        >>> import matplotlib.pyplot as plt
        >>> from grama.models import make_cantilever_beam
        >>> md = make_cantilever_beam()
        >>> md >> \
        >>>     gr.ev_sinews(df_det="swp", skip=True) >> \
        >>>     gr.pt_sinew_inputs(var=md.var)
        >>> plt.show()

    """
    if var is None:
        raise ValueError("Must provide input columns list as keyword var")

    ## Plot
    return pairplot(data=df, vars=var, hue=sweep_ind)


@pipe
def pt_sinew_inputs(*args, **kwargs):
    return plot_sinew_inputs(*args, **kwargs)


@curry
def plot_sinew_outputs(
    df, var=None, out=None, sweep_ind="sweep_ind", sweep_var="sweep_var"
):
    r"""Construct sinew plot

    Create a relational lineplot with hues. Often used to visualize the outputs
    of a sinew design.

    Args:
        df (Pandas DataFrame): Input design data with output results
        var (list of strings): Variables to plot
        out (list of strings): Outputs to plot
        sweep_ind (string): Sweep index column in df
        sweep_var (string): Swept variable column in df

    Returns:
        Seaborn relational lineplot

    Examples:

        >>> import grama as gr
        >>> import matplotlib.pyplot as plt
        >>> from grama.models import make_cantilever_beam
        >>> md = make_cantilever_beam()
        >>> md >> \
        >>>     gr.ev_sinews(df_det="swp") >> \
        >>>     gr.pt_sinew_inputs(var=md.var, out=md.out)
        >>> plt.show()

    """
    if var is None:
        raise ValueError("Must provide input columns list as keyword arg var")
    if out is None:
        raise ValueError("Must provide output columns list as keyword arg out")

    ## Prepare data
    # Gather inputs
    id_vars = [col for col in df.columns if col not in var]
    df_tmp = melt(df, id_vars, var, "_var", "_x")

    # Gather outputs
    id_vars = [col for col in df_tmp.columns if col not in out]
    df_plot = melt(df_tmp, id_vars, out, "_out", "_y")

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
        facet_kws=dict(sharex=False, sharey=False),
    )


@pipe
def pt_sinew_outputs(*args, **kwargs):
    return plot_sinew_outputs(*args, **kwargs)


## Autoplot dispatcher
## ##################################################
plot_list = {
    "sinew_inputs": plot_sinew_inputs,
    "sinew_outputs": plot_sinew_outputs,
    "monte_carlo_inputs": plot_scattermat,
    "monte_carlo_outputs": plot_hists,
}


@curry
def plot_auto(df):
    r"""Automagic plotting

    Convenience tool for various grama outputs. Prints delegated plotting
    function, which can be called manually with different arguments for
    more tailored plotting.

    Args:
        df (DataFrame): Data output from appropriate grama routine. See
            gr.plot_list.keys() for list of supported methods.

    Returns:
        Plot results

    """
    try:
        d = df._plot_info
    except AttributeError:
        raise AttributeError(
            "'{}' object has no attribute _plot_info. Use plot_auto() for grama outputs only.".format(
                "DataFrame"
            )
        )

    try:
        plot_fcn = plot_list[d["type"]]
    except KeyError:
        raise KeyError("'{}' Plot type not implemented.".format(d["type"]))
    plt_kwargs = {key: value for key, value in d.items() if key != "type"}

    print("Calling {0:}....".format(plot_fcn.__name__))

    return plot_fcn(df, **plt_kwargs)


@pipe
def pt_auto(*args, **kwargs):
    return plot_auto(*args, **kwargs)
