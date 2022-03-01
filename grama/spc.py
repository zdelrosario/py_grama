# Statistical Process Control tools
__all__ = [
    "plot_xbs",
    "pt_xbs",
]

from scipy.special import gamma
from grama import add_pipe, Intention
from grama import tf_group_by, tf_summarize, tf_mutate, tf_ungroup, tf_filter
from grama import tf_pivot_longer, tf_left_join
from grama import mean, sd, lead, lag, consec
from grama import n as nfcn
from plotnine import ggplot, aes, geom_line, geom_hline, geom_point, facet_grid, scale_linetype_manual
from plotnine import labs, labeller
from numpy import sqrt

## Helper functions
# --------------------------------------------------
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


## Control Chart constructors
# --------------------------------------------------
def plot_xbs(df, group, var, n_side=9, n_delta=6):
    r"""Construct Xbar and S chart

    Construct an Xbar and S chart to assess the state of statistical control of
    a dataset.

    Args:
        df (DataFrame): Data to analyze
        group (str): Variable for grouping
        var (str): Variable to study

    Keyword args:
        n_side (int): Number of consecutive runs above/below centerline to flag
        n_delta (int): Number of consecutive runs increasing/decreasing to flag

    Returns:
        plotnine object: Xbar and S chart

    Examples:

        import grama as gr
        DF = gr.Intention()

        from grama.data import df_shewhart
        (
            df_shewhart
            >> gr.tf_mutate(idx=DF.index // 10)
            >> gr.pt_xbs("idx", "tensile_strength")
        )

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
            X_mu=mean(DF.X),
            S_biased=mean(DF.S),
            n=mean(DF.n),
        )
    )
    n = df_stats.n[0]
    df_stats["S_mu"] = df_stats.S_biased / c_sd(n)
    df_stats["X_LCL"] = df_stats.X_mu - 3 * df_stats.S_mu / sqrt(n)
    df_stats["X_UCL"] = df_stats.X_mu + 3 * df_stats.S_mu / sqrt(n)
    df_stats["S_LCL"] = B3(n) * df_stats.S_mu
    df_stats["S_UCL"] = B4(n) * df_stats.S_mu

    ## Reshape for plotting
    df_stats_long = (
        df_stats
        >> tf_pivot_longer(
            columns=["X_LCL", "X_mu", "X_UCL", "S_LCL", "S_mu", "S_UCL"],
            names_to=["var", "stat"],
            names_sep="_",
            values_to="value",
        )
    )
    df_batched_long = (
        df_batched
        >> tf_pivot_longer(
            columns=["X", "S"],
            names_to="var",
            values_to="value",
        )
        ## Flag patterns
        >> tf_left_join(
            df_stats
            >> tf_pivot_longer(
                columns=["X_LCL", "X_mu", "X_UCL", "S_LCL", "S_mu", "S_UCL"],
                names_to=["var", ".value"],
                names_sep="_",
            ),
            by="var",
        )
        >> tf_group_by("var")
        >> tf_mutate(
            outlier=(DF.value < DF.LCL) | (DF.UCL < DF.value), # Outside control limits
            below=consec(DF.value < DF.mu, i=n_side), # Below mean
            above=consec(DF.mu < DF.value, i=n_side), # Above mean
        )
        >> tf_mutate(
            decreasing=consec((lead(DF.value) - DF.value) < 0, i=n_delta-1) | # Decreasing
                       consec((DF.value - lag(DF.value)) < 0, i=n_delta-1),
            increasing=consec(0 < (lead(DF.value) - DF.value), i=n_delta-1) | # Increasing
                       consec(0 < (DF.value - lag(DF.value)), i=n_delta-1),
        )
        >> tf_ungroup()
    )

    ## Visualize
    return (
        df_batched_long
        >> ggplot(aes(group))
        + geom_hline(
            data=df_stats_long,
            mapping=aes(yintercept="value", linetype="stat"),
        )
        + geom_line(aes(y="value"), size=0.2)
        + geom_point(aes(y="value"), size=1)
        + geom_point(
            data=df_batched_long
            >> tf_filter(DF.below | DF.decreasing),
            mapping=aes(y="value"),
            color="cyan",
            size=2,
        )
        + geom_point(
            data=df_batched_long
            >> tf_filter(DF.above | DF.increasing),
            mapping=aes(y="value"),
            color="salmon",
            size=2,
        )
        + geom_point(
            data=df_batched_long
            >> tf_filter(DF.outlier),
            mapping=aes(y="value"),
            color="red",
            size=2,
        )
        + scale_linetype_manual(
            values=dict(LCL="dashed", UCL="dashed", mu="solid")
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
