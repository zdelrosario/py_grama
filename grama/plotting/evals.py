__all__ = [
    "plot_monte_carlo_inputs",
    "pt_monte_carlo_inputs",
    "plot_monte_carlo_outputs",
    "pt_monte_carlo_outputs"
]

import pandas as pd
from ..tools import pipe
from ..transforms import tran_gather
from toolz import curry
import seaborn as sns
import matplotlib.pyplot as plt

## Monte Carlo
# --------------------------------------------------
@curry
def plot_monte_carlo_inputs(df, var_rand=None):
    """Inspect the design
    """
    if var_rand is None:
        raise ValueError("Must provide input columns list as keyword var_rand")

    ## Plot
    return sns.pairplot(data=df, vars=var_rand)

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
    df_gathered = tran_gather(df, "key", "out", out)

    ## Faceted histograms
    g = sns.FacetGrid(df_gathered, col="key", sharex=False, sharey=False)
    g.map(plt.hist, "out")

    return g

@pipe
def pt_monte_carlo_outputs(*args, **kwargs):
    return plot_monte_carlo_outputs(*args, **kwargs)
