__all__ = [
    "plot_auto",
    "pt_auto",
    "plot_list"
]

from ..tools import pipe
from toolz import curry

from .sweeps import plot_sinew_inputs, plot_sinew_outputs

plot_list = {
    "sinew_inputs": plot_sinew_inputs,
    "sinew_outputs": plot_sinew_outputs
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
        raise AttributeError("'{}' object has no attribute _plot_info\n  Use plot_auto() for grama outputs only.".format('DataFrame'))

    try:
        plot_fcn = plot_list[d["type"]]
    except KeyError:
        raise KeyError("'{}'\n  Plot type not implemented.")
    plt_kwargs = {key: value for key, value in d.items() if key != "type"}

    return plot_fcn(df, **plt_kwargs)

@pipe
def pt_auto(*args, **kwargs):
    plot_auto(*args, **kwargs)
