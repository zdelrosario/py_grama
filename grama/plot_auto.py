__all__ = [
    "plot_contour",
    "pt_contour",
    "plot_corrtile",
    "pt_corrtile",
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

from grama import add_pipe, pipe, tf_pivot_longer, tf_outer, tf_select, tf_rename, tf_filter
from grama import Intention
from pandas import melt

from plotnine import aes, annotate, ggplot, facet_grid, facet_wrap, labs, element_text, guides
from plotnine import theme, theme_void, theme_minimal
from plotnine import scale_x_continuous, scale_y_continuous, scale_fill_gradient2
from plotnine import geom_point, geom_density, geom_histogram, geom_line, geom_tile
from plotnine import geom_segment, geom_blank
from matplotlib import gridspec

from toolz import curry

## Helper functions
##################################################
def _sci_format(v):
    r"""Scientific format
    """
    return (
        ["{0:1.1e}".format(v[0])] +
        [""] * (len(v) - 2) +
        ["\n{0:1.1e}".format(v[-1])]
    )

## Function-specific plot functions
##################################################
## eval_contour
@curry
def plot_contour(df, var=None, out="out", level="level", aux=False):
    r"""Plot 2d contours

    Plot contours.

    Usually called as a dispatch from plot_auto().

    Args:
        var (array of str): Variables for plot axes
        out (str): Name of output identifier column
        level (str): Name of level identifier column
        aux (bool): Auxillary variables present?

    Returns:
        ggplot: Contour image

    Examples:

        >>> import grama as gr
        >>> from grama.models import make_cantilever_beam

    """
    # Check invariants
    if var is None:
        raise ValueError("Must provide input columns list as keyword var")
    if aux:
        raise ValueError(
            "Autoplot plot_contour not designed to handle auxiliary variables. " +
            "Regenerate contour data with fixed auxilary variables, " +
            "or try creating a manual plot."
        )

    return (
        df
        >> ggplot()
        + geom_segment(
            aes(
                var[0],
                var[1],
                xend=var[0]+"_end",
                yend=var[1]+"_end",
                linetype=out,
                color=level,
            )
        )
    )


pt_contour = add_pipe(plot_contour)

## tran_iocorr
# --------------------------------------------------
@curry
def plot_corrtile(df, var=None, out=None, corr=None):
    r"""
    """
    return (
        df
        >> ggplot(aes(var, out))
        + geom_tile(aes(fill=corr))
        + scale_fill_gradient2(name="Corr", midpoint=0)
        + theme(axis_text_x=element_text(angle=270))
    )

pt_corrtile = add_pipe(plot_corrtile)

## Sample
# --------------------------------------------------
@curry
def plot_scattermat(df, var=None):
    r"""Create a scatterplot matrix

    Create a scatterplot matrix. Often used to visualize a design (set of inputs
    points) before evaluating the functions.

    Usually called as a dispatch from plot_auto().

    Args:
        var (list of strings): Variables to plot

    Returns:
        ggplot: Scatterplot matrix

    Examples:

        >>> import grama as gr
        >>> import matplotlib.pyplot as plt
        >>> from grama.models import make_cantilever_beam
        >>> md = make_cantilever_beam()
        >>> ## Dispatch from autoplotter
        >>> (
        >>>     md
        >>>     >> gr.ev_sample(n=100, df_det="nom", skip=True)
        >>>     >> gr.pt_auto()
        >>> )
        >>> ## Re-create plot without metadata
        >>> (
        >>>     md
        >>>     >> gr.ev_sample(n=100, df_det="nom")
        >>>     >> gr.pt_scattermat(var=md.var)
        >>> )

    """
    if var is None:
        raise ValueError("Must provide input columns list as keyword var")

    ## Define helpers
    labels_blank = lambda v: [""] * len(v)
    breaks_min = lambda lims: (lims[0], 0.5 * (lims[0] + lims[1]), lims[1])

    ## Make blank figure
    fig = (
        df
        >> ggplot()
        + geom_blank()
        + theme_void()
    ).draw(show=False)

    gs = gridspec.GridSpec(len(var), len(var))
    for i, v1 in enumerate(var):
        for j, v2 in enumerate(var):
            ax = fig.add_subplot(gs[i, j])
            ## Switch labels
            if j == 0:
                labels_y = _sci_format
            else:
                labels_y = labels_blank
            if i == len(var) - 1:
                labels_x = _sci_format
            else:
                labels_x = labels_blank

            ## Density
            if i == j:
                xmid = 0.5 * (
                    df[v1].min() + df[v1].max()
                )

                p = (
                    df
                    >> ggplot(aes(v1))
                    + geom_density()
                    + scale_x_continuous(
                        breaks=breaks_min,
                        labels=labels_x,
                    )
                    + scale_y_continuous(
                        breaks=breaks_min,
                        labels=labels_y,
                    )
                    + annotate(
                        "label",
                        x=xmid,
                        y=0,
                        label=v1,
                        va="bottom",
                    )
                    + theme_minimal()
                    + labs(title=v1)
                )

            ## Scatterplot
            else:
                p = (
                    df
                    >> ggplot(aes(v2, v1))
                    + geom_point()
                    + scale_x_continuous(
                        breaks=breaks_min,
                        labels=labels_x,
                    )
                    + scale_y_continuous(
                        breaks=breaks_min,
                        labels=labels_y,
                    )
                    + theme_minimal()
                    + theme(
                        axis_title=element_text(va="top", size=12),
                    )
                )

            _ = p._draw_using_figure(fig, [ax])


    ## Plot
    # NB Returning the figure causes a "double plot" in Jupyter....
    fig.show()


pt_scattermat = add_pipe(plot_scattermat)


@curry
def plot_hists(df, out=None, **kwargs):
    r"""Construct histograms

    Create a set of histograms. Often used to visualize the results of random
    sampling for multiple outputs.

    Usually called as a dispatch from plot_auto().

    Args:
        out (list of strings): Variables to plot

    Returns:
        Seaborn histogram plot

    Examples:

        >>> import grama as gr
        >>> import matplotlib.pyplot as plt
        >>> from grama.models import make_cantilever_beam
        >>> md = make_cantilever_beam()
        >>> ## Dispatch from autoplotter
        >>> (
        >>>     md
        >>>     >> gr.ev_sample(n=100, df_det="nom")
        >>>     >> gr.pt_auto()
        >>> )
        >>> ## Re-create without metadata
        >>> (
        >>>     md
        >>>     >> gr.ev_sample(n=100, df_det="nom")
        >>>     >> gr.pt_hists(out=md.out)
        >>> )

    """
    if out is None:
        raise ValueError("Must provide input columns list as keyword out")

    return (
        df
        >> tf_pivot_longer(
            columns=out,
            names_to="var",
            values_to="value",
        )
        >> ggplot(aes("value"))
        + geom_histogram(bins=30)
        + facet_wrap("var", scales="free")
        + theme_minimal()
        + labs(
            x="Output Value",
            y="Count",
        )
    )


pt_hists = add_pipe(plot_hists)

## Sinew plots
# --------------------------------------------------
@curry
def plot_sinew_inputs(df, var=None, sweep_ind="sweep_ind"):
    r"""Inspect a sinew design

    Create a scatterplot matrix with hues. Often used to visualize a sinew
    design before evaluating the model functions.

    Usually called as a dispatch from plot_auto().

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
        >>> ## Dispatch from autoplotter
        >>> (
        >>>     md
        >>>     >> gr.ev_sinews(df_det="swp", skip=True)
        >>>     >> gr.pt_auto()
        >>> )
        >>> ## Re-create without metadata
        >>> (
        >>>     md
        >>>     >> gr.ev_sinews(df_det="swp")
        >>>     >> gr.pt_sinew_inputs(var=md.var)
        >>> )

    """
    if var is None:
        raise ValueError("Must provide input columns list as keyword var")

    ## Define helpers
    labels_blank = lambda v: [""] * len(v)
    breaks_min = lambda lims: (lims[0], 0.5 * (lims[0] + lims[1]), lims[1])

    ## Make blank figure
    fig = (
        df
        >> ggplot()
        + geom_blank()
        + theme_void()
    ).draw(show=False)

    gs = gridspec.GridSpec(len(var), len(var))
    for i, v1 in enumerate(var):
        for j, v2 in enumerate(var):
            ax = fig.add_subplot(gs[i, j])
            ## Switch labels
            if j == 0:
                labels_y = _sci_format
            else:
                labels_y = labels_blank
            if i == len(var) - 1:
                labels_x = _sci_format
            else:
                labels_x = labels_blank

            ## Label
            if i == j:
                p = (
                    df
                    >> ggplot()
                    + annotate(
                        "label",
                        x=0,
                        y=0,
                        label=v1,
                    )
                    + theme_void()
                    + guides(color=None)
                )

            ## Scatterplot
            else:
                p = (
                    df
                    >> ggplot(aes(v2, v1, color="factor("+sweep_ind+")"))
                    + geom_point(size=0.1)
                    + scale_x_continuous(
                        breaks=breaks_min,
                        labels=labels_x,
                    )
                    + scale_y_continuous(
                        breaks=breaks_min,
                        labels=labels_y,
                    )
                    + guides(color=None)
                    + theme_minimal()
                    + theme(
                        axis_title=element_text(va="top", size=12),
                    )
                )

            _ = p._draw_using_figure(fig, [ax])


    ## Plot
    # NB Returning the figure causes a "double plot" in Jupyter....
    fig.show()


pt_sinew_inputs = add_pipe(plot_sinew_inputs)


@curry
def plot_sinew_outputs(
    df, var=None, out=None, sweep_ind="sweep_ind", sweep_var="sweep_var"
):
    r"""Construct sinew plot

    Create a relational lineplot with hues. Often used to visualize the outputs
    of a sinew design.

    Usually called as a dispatch from plot_auto().

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
        >>> ## Dispatch from autoplotter
        >>> (
        >>>     md
        >>>     >> gr.ev_sinews(df_det="swp")
        >>>     >> gr.pt_auto()
        >>> )
        >>> ## Re-create without metadata
        >>> (
        >>>     md
        >>>     >> gr.ev_sinews(df_det="swp")
        >>>     >> gr.pt_sinew_inputs(var=md.var, out=md.out)
        >>> )

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

    breaks_min = lambda lims: (lims[0], 0.5 * (lims[0] + lims[1]), lims[1])
    return (
        df_plot
        >> ggplot(aes(
            "_x",
            "_y",
            color="factor(" + sweep_ind + ")",
            group="factor(" + sweep_ind + ")",
        ))
        + geom_line()
        + facet_grid("_out~_var", scales="free")

        + scale_x_continuous(
            breaks=breaks_min,
            labels=_sci_format,
        )
        + scale_y_continuous(
            breaks=breaks_min,
            labels=_sci_format,
        )
        + guides(color=None)
        + theme_minimal()
        + theme(strip_text_y=element_text(angle=0))
        + labs(
            x="Input Value",
            y="Output Value",
        )
    )


pt_sinew_outputs = add_pipe(plot_sinew_outputs)

## Autoplot dispatcher
## ##################################################
plot_list = {
    "contour": plot_contour,
    "sinew_inputs": plot_sinew_inputs,
    "sinew_outputs": plot_sinew_outputs,
    "sample_inputs": plot_scattermat,
    "sample_outputs": plot_hists,
    "iocorr": plot_corrtile,
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


pt_auto = add_pipe(plot_auto)
