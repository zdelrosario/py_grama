# Statistical Process Control tools
__all__ = [
    "c_sd",
    "B3",
    "B4",
    "plot_xbs",
    "pt_xbs",
]

from scipy.special import gamma
from grama import add_pipe, Intention
from grama import tf_group_by, tf_summarize, tf_mutate, tf_ungroup, tf_filter
from grama import tf_pivot_longer, tf_left_join
from grama import mean, sd, lead, lag, consec, case_when
from grama import n as nfcn
from plotnine import ggplot, aes, geom_line, geom_hline, geom_point, facet_grid, theme, guides
from plotnine import scale_linetype_manual, scale_shape_manual, scale_color_manual
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
            X_center=mean(DF.X),
            S_biased=mean(DF.S),
            n=mean(DF.n),
        )
    )
    n = df_stats.n[0]
    df_stats["S_center"] = df_stats.S_biased / c_sd(n)
    df_stats["X_LCL"] = df_stats.X_center - 3 * df_stats.S_center / sqrt(n)
    df_stats["X_UCL"] = df_stats.X_center + 3 * df_stats.S_center / sqrt(n)
    df_stats["S_LCL"] = B3(n) * df_stats.S_center
    df_stats["S_UCL"] = B4(n) * df_stats.S_center

    ## Reshape for plotting
    df_stats_long = (
        df_stats
        >> tf_pivot_longer(
            columns=["X_LCL", "X_center", "X_UCL", "S_LCL", "S_center", "S_UCL"],
            names_to=["_var", "_stat"],
            names_sep="_",
            values_to="_value",
        )
    )
    # Fake group value to avoid issue with discrete group variable
    df_stats_long[group] = [df_batched[group].values[0]] * df_stats_long.shape[0]

    df_batched_long = (
        df_batched
        >> tf_pivot_longer(
            columns=["X", "S"],
            names_to="_var",
            values_to="_value",
        )
        ## Flag patterns
        >> tf_left_join(
            df_stats
            >> tf_pivot_longer(
                columns=["X_LCL", "X_center", "X_UCL", "S_LCL", "S_center", "S_UCL"],
                names_to=["_var", ".value"],
                names_sep="_",
            ),
            by="_var",
        )
        >> tf_group_by("_var")
        >> tf_mutate(
            outlier_below=(DF._value < DF.LCL),        # Outside control limits
            outlier_above=(DF.UCL < DF._value),
            below=consec(DF._value < DF.center, i=n_side), # Below mean
            above=consec(DF.center < DF._value, i=n_side), # Above mean
        )
        >> tf_mutate(
            decreasing=consec((lead(DF._value) - DF._value) < 0, i=n_delta-1) | # Decreasing
                       consec((DF._value - lag(DF._value)) < 0, i=n_delta-1),
            increasing=consec(0 < (lead(DF._value) - DF._value), i=n_delta-1) | # Increasing
                       consec(0 < (DF._value - lag(DF._value)), i=n_delta-1),
        )
        >> tf_mutate(
            sign=case_when(
                [DF.outlier_below, "-2"],
                [DF.outlier_above, "+2"],
                [DF.below | DF.decreasing, "-1"],
                [DF.above | DF.increasing, "+1"],
                [True, "0"]
            ),
            glyph=case_when(
                [DF.outlier_below, "Below Limit"],
                [DF.outlier_above, "Above Limit"],
                [DF.below, "Low Run"],
                [DF.above, "High Run"],
                [DF.increasing, "Increasing Run"],
                [DF.decreasing, "Decreasing Run"],
                [True, "None"],
            )
        )
        >> tf_ungroup()
    )

    ## Visualize
    return (
        df_batched_long
        >> ggplot(aes(x=group))
        + geom_hline(
            data=df_stats_long,
            mapping=aes(yintercept="_value", linetype="_stat"),
        )
        + geom_line(aes(y="_value", group="_var"), size=0.2)
        + geom_point(
            aes(y="_value", color="sign", shape="glyph"),
            size=3,
        )

        + scale_color_manual(
            values={"-2": "blue", "-1": "darkturquoise", "0": "black", "+1": "salmon", "+2": "red"},
        )
        + scale_shape_manual(
            name="Patterns",
            values={
                "Below Limit": "s",
                "Above Limit": "s",
                "Low Run": "X",
                "High Run": "X",
                "Increasing Run": "^",
                "Decreasing Run": "v",
                "None": "."
            },
        )
        + scale_linetype_manual(
            name="Guideline",
            values=dict(LCL="dashed", UCL="dashed", center="solid"),
        )
        + guides(color=None)
        + facet_grid(
            "_var~.",
            scales="free_y",
            labeller=labeller(dict(X="Mean", S="Variability")),
        )
        + labs(
            x="Group variable ({})".format(group),
            y="Value ({})".format(var),
        )
    )


pt_xbs = add_pipe(plot_xbs)
