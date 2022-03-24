__all__ = [
    "tran_polyridge",
    "tf_polyridge",
    "fit_polyridge",
    "ft_polyridge",
]

import copy
from grama import add_pipe, Function, Model
from numpy import number as npnumber
from pandas import DataFrame
from toolz import curry

from .polyridge import PolynomialRidgeApproximation

## Helper functions
# --------------------------------------------------
class FunctionPoly(Function):
    def __init__(self, regressor, var, out, name, runtime):
        """

        Args:
            regressor (scikit-style Regressor):
        """
        self.regressor = regressor
        self.var = var
        self.out = list(map(lambda s: s + "_mean", out))
        self.name = name
        self.runtime = runtime

    def copy(self):
        """Make a copy"""
        func_new = FunctionPoly(
            copy.deepcopy(self.regressor),
            copy.deepcopy(self.var),
            copy.deepcopy(self.out),
            copy.deepcopy(self.name),
            runtime=self.runtime,
        )
        return func_new

    def eval(self, df):
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.var).issubset(set(df.columns)):
            raise ValueError(
                "Model function `{}` var not a subset of given columns".format(
                    self.name
                )
            )

        ## Predict
        y = self.regressor.predict(df[self.var])
        return DataFrame(data=y, columns=self.out)


## Implementation
# --------------------------------------------------
def _polyridge(
    df, var=None, out=None, n_dim=None, n_degree=None, **kwargs
):
    r"""Low-level interface for polyridge

    Apply the polynomial ridge approximation to seek a low-dimensional subspace
    among inputs that best explain variability in a selected output. Return the
    scikit-style fitted object and modified `var`.

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
        PolynomialRidgeApproximation: Scikit-style regressor
        list of str: Modified variable list

    Notes:
        - A wrapper for PolynomialRidgeApproximation, originally implemented in the psdr package.

    References:
        - J. M. Hokanson and Paul G. Constantine. Data-driven Polynomial Ridge Approximation Using Variable Projection. SIAM J. Sci. Comput. Vol 40, No 3, pp A1566–A1589, DOI:10.1137/17M1117690.

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

    return pr, var

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
        - A wrapper for PolynomialRidgeApproximation, originally implemented in the psdr package.

    References:
        - J. M. Hokanson and Paul G. Constantine. Data-driven Polynomial Ridge Approximation Using Variable Projection. SIAM J. Sci. Comput. Vol 40, No 3, pp A1566–A1589, DOI:10.1137/17M1117690.

    Examples:
        import grama as gr
        from grama.psdr import tf_polyridge
        DF = gr.Intention()

        ## Set up a dataset to reduce
        df_data = (
            gr.df_make(x=range(10))
            >> gr.tf_outer(gr.df_make(y=range(10)))
            >> gr.tf_outer(gr.df_make(z=range(10)))
            >> gr.tf_mutate(f=DF.x - DF.y)
        )

        ## Use polynomial ridge approximation to derive weights
        df_weights = (
            df_data
            >> tf_polyridge(out="f", n_dim=1, n_degree=1)
        )

        ## Construct a shadow plot
        (
            df_data
            >> gr.tf_inner(df_weights)
            >> gr.ggplot(gr.aes("dot", "f"))
            + gr.geom_point()
        )

    """
    ## Run the low-level implementation
    pr, var = _polyridge(df, var=var, out=out, n_dim=n_dim, n_degree=n_degree, **kwargs)

    ## Package the results
    df_res = DataFrame(
        data=pr.U.T,
        columns=var
    )
    return df_res

tf_polyridge = add_pipe(tran_polyridge)

@curry
def fit_polyridge(
    df, var=None, out=None, n_dim=None, n_degree=None, domain=None, density=None, **kwargs
):
    r"""Polynomial Ridge Approximation fitting routine

    Fit a polynomial model on a subspace reduction of the given variables. Uses
    the variable projection (L2) algorithm of Hokanson and Constantine (2018).

    Args:
        df (DataFrame): Target dataset to reduce
        var (list or None): Variables in df on which to perform dimension reduction.
            Use None to compute with all numeric variables, except `out`.
        out (string): Name of target output
        domain (gr.Domain or None): Domain for fitted model, optional
        density (gr.Density or None): Density for fitted model, optional
        n_dim (int): Target dimensionality
        n_degree (int): Fitted polynomial (total) order

    Kwargs:
        n_init (int): Number of iterations in optimization

    Returns:
        Model: Polynomial (surrogate) model.

    Notes:
        - A wrapper for PolynomialRidgeApproximation, originally implemented in the psdr package.

    References:
        - J. M. Hokanson and Paul G. Constantine. Data-driven Polynomial Ridge Approximation Using Variable Projection. SIAM J. Sci. Comput. Vol 40, No 3, pp A1566–A1589, DOI:10.1137/17M1117690.

    Examples:
        import grama as gr
        DF = gr.Intention()

        ## Set up a dataset to reduce
        df_data = (
            gr.df_make(x=range(10))
            >> gr.tf_outer(gr.df_make(y=range(10)))
            >> gr.tf_outer(gr.df_make(z=range(10)))
            >> gr.tf_mutate(f=DF.x - DF.y)
        )

        ## Use polynomial ridge approximation to fit a surrogate model
        md_poly = (
            df_data
            >> gr.ft_polyridge(out="f", n_dim=1, n_degree=1)
        )

        ## Fit routine is useful for cross-validation of hyperparameters;
        # compare two different hyperparameter settings
        (
            df_data
            >> gr.tf_kfolds(
                ft=gr.ft_polyridge(out="f", n_dim=1, n_degree=1),
                k=5,
            )
        )

        (
            df_data
            >> gr.tf_kfolds(
                ft=gr.ft_polyridge(out="f", n_dim=2, n_degree=2),
                k=5,
            )
        )

    """
    ## Run the low-level implementation
    pr, var = _polyridge(df, var=var, out=out, n_dim=n_dim, n_degree=n_degree, **kwargs)

    ## Package the results
    md_res = Model(
        functions=[FunctionPoly(pr, var, [out], "Polyridge ({})".format(out), None)],
        domain=domain,
        density=density,
    )
    return md_res

ft_polyridge = add_pipe(fit_polyridge)
