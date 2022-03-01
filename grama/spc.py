# Statistical Process Control tools
__all__ = [
    "plot_xbs",
    "pt_xbs",
]

from scipy.special import gamma
from grama import add_pipe, Intention
from grama import tf_group_by, tf_summarize, tf_mutate, tf_ungroup, tf_pivot_longer
from grama import mean, sd
from grama import n as nfcn
from plotnine import ggplot, aes, geom_hline, geom_point, facet_grid, scale_linetype_manual
from plotnine import labs, labeller
from numpy import sqrt

def c_sd(n):
    r"""Anti-biasing constant for aggregate standard deviation

    Returns the anti-biasing constant for aggregated standard deviation
    estimates. If the average of $k$ samples each size $n$ are averaged to
    produce $\overline{S} = (1/k) \sum_{i=1}^k S_i$, then the de-biased standard
    deviation is:

        $$\hat{\sigma} = \overline{S} / c(n)$$

    Arguments:
        n (int): Sample (batch) size

    Returns:
        float: anti-biasing constant

    References:
        Kenett and Zacks, Modern Industrial Statistics (2014) 2nd Ed

    """
    return gamma(n/2) / gamma( (n-1)/2 ) * sqrt( 2 / (n-1) )

def B3(n):
    r"""Lower Control Limit constant for standard deviation

    Returns the Lower Control Limit (LCL) constant for aggregated standard
    deviation estimates. If the average of $k$ samples each size $n$ are
    averaged to produce $\overline{S} = (1/k) \sum_{i=1}^k S_i$, then the LCL
    is:

        $$LCL = B_3 \overline{S}$$

    Arguments:
        n (int): Sample (batch) size

    Returns:
        float: LCL constant

    References:
        Kenett and Zacks, Modern Industrial Statistics (2014) 2nd Ed, Equation (8.22)

    """
    return max( 1 - 3 / c_sd(n) * sqrt(1 - c_sd(n)**2), 0 )

def B4(n):
    r"""Upper Control Limit constant for standard deviation

    Returns the Upper Control Limit (UCL) constant for aggregated standard
    deviation estimates. If the average of $k$ samples each size $n$ are
    averaged to produce $\overline{S} = (1/k) \sum_{i=1}^k S_i$, then the UCL
    is:

        $$UCL = B_4 \overline{S}$$

    Arguments:
        n (int): Sample (batch) size

    Returns:
        float: UCL constant

    References:
        Kenett and Zacks, Modern Industrial Statistics (2014) 2nd Ed, Equation (8.22)

    """
    return 1 + 3 / c_sd(n) * sqrt(1 - c_sd(n)**2)


def plot_xbs(df, group, var):
    r"""Construct Xbar and S chart

    Args:
        df (DataFrame):
        group (str):
        var (str):
    """
    ## Prepare the data
    DF = Intention()
    df_batched = (
        df
        >> tf_group_by(group)
        >> tf_summarize(
            X=mean(DF[var]),
            S=sd(DF[var]),
            n=nfcn(DF.index),
        )
        >> tf_ungroup()
    )

    df_stats = (
        df_batched
        >> tf_summarize(
            X_mean=mean(DF.X),
            S_biased=mean(DF.S),
            n=mean(DF.n),
        )
    )
    n = df_stats.n[0]
    df_stats["S_mean"] = df_stats.S_biased / c_sd(n)
    df_stats["X_LCL"] = df_stats.X_mean - 3 * df_stats.S_mean / sqrt(n)
    df_stats["X_UCL"] = df_stats.X_mean + 3 * df_stats.S_mean / sqrt(n)
    df_stats["S_LCL"] = B3(n) * df_stats.S_mean
    df_stats["S_UCL"] = B4(n) * df_stats.S_mean

    ## Reshape for plotting
    df_batched = (
        df_batched
        >> tf_pivot_longer(
            columns=["X", "S"],
            names_to="var",
            values_to="value",
        )
    )
    df_stats = (
        df_stats
        >> tf_pivot_longer(
            columns=["X_LCL", "X_mean", "X_UCL", "S_LCL", "S_mean", "S_UCL"],
            names_to=["var", "stat"],
            names_sep="_",
            values_to="value"
        )
    )

    ## Flag patterns
    # TODO

    ## Visualize
    return (
        df_batched
        >> ggplot(aes(group))
        + geom_hline(
            data=df_stats,
            mapping=aes(yintercept="value", linetype="stat"),
        )
        + geom_point(aes(y="value"))
        + scale_linetype_manual(
            values=dict(LCL="dashed", UCL="dashed", mean="solid")
        )
        + facet_grid(
            "var~.",
            scales="free_y",
            labeller=labeller(dict(X="Mean", S="Variability")),
        )
        + labs(
            x="Group variable ({})".format(group),
            y="Value ({})".format(var),
        )
    )


pt_xbs = add_pipe(plot_xbs)
