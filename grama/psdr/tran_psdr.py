__all__ = [
    "tran_polyridge",
    "tf_polyridge",
]

from grama import add_pipe
from numpy import number as npnumber
from pandas import DataFrame
from toolz import curry

from .polyridge import PolynomialRidgeApproximation

## Implementation
# --------------------------------------------------

## Interfaces
# --------------------------------------------------
@curry
def tran_polyridge(
    df, var=None, out=None, n_dim=None, n_degree=None, **kwargs
):
    r"""Polynomial Ridge Approximation for parameter space reduction

    Apply the polynomial ridge approximation to seek a low-dimensional subspace
    among inputs that best explain variability in a selected output.

    Args:
        df (DataFrame): Target dataset to reduce
        var (list or None): Variables in df on which to perform dimension reduction.
            Use None to compute with all numeric variables, except `out`.
        out (string): Name of target output
        n_dim (int): Target dimensionality
        n_degree (int): Fitted polynomial (total) order

    Kwargs:
        n_init (int): Number of iterations in optimization

    Returns:
        DataFrame: Set of weights defining the dimension-reduced space.

    Notes:
        - A wrapper for psdr.PolynomialRidgeApproximation

    References:
        - J. M. Hokanson and Paul G. Constantine. Data-driven Polynomial Ridge Approximation Using Variable Projection. SIAM J. Sci. Comput. Vol 40, No 3, pp A1566–A1589, DOI:10.1137/17M1117690.

    Examples:
        >>> import grama as gr
        >>> from plotnine import *
        >>> from grama.psdr import tf_polyridge
        >>> DF = gr.Intention()
        >>>
        >>> ## Set up a dataset to reduce
        >>> df_data = (
        >>>     gr.df_make(x=range(10))
        >>>     >> gr.tf_outer(gr.df_make(y=range(10)))
        >>>     >> gr.tf_outer(gr.df_make(z=range(10)))
        >>>     >> gr.tf_mutate(f=DF.x - DF.y)
        >>> )
        >>>
        >>> ## Use polynomial ridge approximation to derive weights
        >>> df_weights = (
        >>>     df_data
        >>>     >> tf_polyridge(out="f", n_dim=1, n_degree=1)
        >>> )
        >>>
        >>> ## Construct a shadow plot
        >>> (
        >>>     df_data
        >>>     >> gr.tf_inner(df_weights)
        >>>     >> ggplot(aes("dot", "f"))
        >>>     + geom_point()
        >>> )

    """
    ## Check invariants
    if out is None:
        raise ValueError("Must select an `out` column.")
    if not (out in df.columns):
        raise ValueError("out={} is not a column in provided dataframe.".format(out))

    if var is None:
        var = set(df.select_dtypes(include=[npnumber]).columns.values)
        var = list(var.difference({out}))
    else:
        var = list(var).copy()
        diff = set(var).difference(set(df.columns))
        if len(diff) > 0:
            raise ValueError(
                "`var` must be subset of `df.columns`\n" "diff = {}".format(diff)
            )

    if (n_degree == 1) and (n_dim > 1):
        raise ValueError(
            "n_dim > 1 cannot work with n_degree == 1; this would be " +
            "an over-parameterized linear model. Try again with n_degree > 1."
        )

    ## Compute subspace reduction
    pr = PolynomialRidgeApproximation(
        n_degree,
        n_dim,
        **kwargs,
    )
    pr.fit(df[var].values, df[out].values)

    ## Package the results
    df_res = DataFrame(
        data=pr.U.T,
        columns=var
    )
    return df_res

tf_polyridge = add_pipe(tran_polyridge)
