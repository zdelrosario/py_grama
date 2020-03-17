__all__ = [
    "tran_asub",
    "tf_asub",
    "tran_describe",
    "tf_describe",
    "tran_inner",
    "tf_inner",
    "tran_pca",
    "tf_pca",
    "tran_sobol",
    "tf_sobol",
]

from numpy import round, dot
from numpy.linalg import svd
from pandas import concat, DataFrame

import re
import itertools
import warnings

from grama import pipe, custom_formatwarning
from toolz import curry

warnings.formatwarning = custom_formatwarning

## Compute Sobol' indices
# --------------------------------------------------
@curry
def tran_sobol(df, typename="ind", digits=2, full=False):
    r"""Post-process results from gr.eval_hybrid()

    Estimate Sobol' indices based on hybrid point evaluations (Sobol', 1999).
    Intended as post-processor for gr.eval_hybrid().

    Args:
        df (DataFrame): Hybrid point results from gr.eval_hybrid()
        typename (str): Name to give index type column in results
        digits (int): Number of digits for rounding reported results
        full (bool): Return un-normalized indices and variance?

    Returns:
        DataFrame: Sobol' indices

    Notes:
        - Index type ["first", "total"] is inferred from input df._meta;
          this is assigned by gr.eval_hybrid().
        - Index normalization coded in the "ind" column;
          S: Normalized index
          T: Un-normalized index
          var: Total variance

    References:
        I.M. Sobol', "Sensitivity Estimates for Nonlinear Mathematical Models"
        (1999) MMCE, Vol 1.

    Examples:

        >>> import grama as gr
        >>> from grama.models import make_cantilever_beam
        >>> md = make_cantilever_beam()
        >>> df_first = md >> gr.ev_hybrid(df_det="nom", plan="first")
        >>> df_first >> gr.tf_sobol()
        >>>
        >>> df_total = md >> gr.ev_hybrid(df_det="nom", plan="total")
        >>> df_total >> gr.tf_sobol()

    """
    ## Determine plan from dataframe metadata
    metadata = df._meta
    if metadata["type"] == "eval_hybrid":
        plan = metadata["plan"]
        varname = metadata["varname"]
        var_rand = metadata["var_rand"]
        out = metadata["out"]
    else:
        raise ValueError("df not hybrid points!")

    ## Check invariants
    if not (varname in df.columns):
        raise ValueError("{} not in df.columns".format(varname))

    ## Setup
    I_base = df[varname] == "_"
    df_var = DataFrame(df[out].var()).transpose()
    df_var[typename] = "var"

    df_res = df_var.copy()

    if plan == "first":
        mu_base = df[out][I_base].mean()

        for i_var, var in enumerate(var_rand):
            I_var = df[varname] == var

            mu_var = df[out][I_var].mean()
            mu_tot = 0.5 * (mu_base + mu_var)
            s2_var = (
                df[out][I_base]
                .reset_index(drop=True)
                .mul(df[out][I_var].reset_index(drop=True))
            ).mean()

            df_tau = DataFrame(s2_var - mu_tot ** 2).transpose()
            df_tau[typename] = "T_" + var

            df_index = (
                df_tau[out]
                .reset_index(drop=True)
                .truediv(df_var.drop(columns=typename))
            )
            df_index[typename] = "S_" + var

            df_res = concat((df_res, df_tau, df_index))

    elif plan == "total":
        for i_var, var in enumerate(var_rand):
            I_var = df[varname] == var
            s2_var = (
                (
                    df[out][I_base].reset_index(drop=True)
                    - df[out][I_var].reset_index(drop=True)
                )
                ** 2
            ).mean() * 0.5

            df_tau = DataFrame(s2_var).transpose()
            df_tau[typename] = "T_" + var

            df_index = (
                df_tau.drop(columns=typename)
                .reset_index(drop=True)
                .truediv(df_var.drop(columns=typename))
            )
            df_index[typename] = "S_" + var

            df_res = concat((df_res, df_tau, df_index))
    else:
        raise ValueError("plan `{}` not valid".format(plan))

    ## Post-process
    outputs = df_res.drop(typename, axis=1).columns
    df_res[outputs] = df_res[outputs].apply(lambda row: round(row, decimals=digits))
    df_res.sort_values(typename, inplace=True)

    ## Filter, if necessary
    if not full:
        I_normalized = list(map(lambda s: s[0] == "S", df_res[typename]))
        df_res = df_res[I_normalized]

    ## Fill NaN's
    df_res.fillna(value=0, inplace=True)

    return df_res


@pipe
def tf_sobol(*args, **kwargs):
    return tran_sobol(*args, **kwargs)


## Linear algebra tools
##################################################
## Principal Component Analysis (PCA)
@curry
def tran_pca(df, var=None, lamvar="lam", standardize=False):
    r"""Principal Component Analysis

    Compute principal directions and eigenvalues for a dataset. Can specify
    columns to analyze, or just analyze all numerical columns.

    Args:
        df (DataFrame): Data to analyze
        var (list of str or None): List of columns to analyze
        lambvar (str): Name to give eigenvalue column; default="lam"
        standardize (bool): Standardize columns? default=False

    Returns:
        DataFrame: principal directions and eigenvalues

    References:
        TODO

    Examples:

        >>> import grama as gr
        >>> from grama.data import df_stang
        >>> df_pca = df_stang >> gr.tf_pca()

    """
    ## Handle variable selection
    columns_numeric = list(df.select_dtypes("number").columns)
    if var is None:
        var = columns_numeric
    else:
        if not set(var).issubset(set(columns_numeric)):
            raise ValueError("`var` must be a subset of numeric df.columns")

    ## Setup
    X = df[var].values
    if standardize:
        U, s, Vh = svd((X - df[var].mean().values) / df[var].std().values)
    else:
        U, s, Vh = svd(X - df[var].mean().values)

    df_tmp = DataFrame(data=Vh, columns=var)
    df_tmp[lamvar] = s

    return df_tmp[[lamvar] + var]


@pipe
def tf_pca(*args, **kwargs):
    return tran_pca(*args, **kwargs)


## Gradient principal directions (AS)
@curry
def tran_asub(df, prefix="D", outvar="out", lamvar="lam"):
    r"""Active subspace estimator

    Compute principal directions and eigenvalues for all outputs based on output
    of ev_grad_fd() to estimate the /active subspace/ (Constantine, 2015).

    Args:
        df (DataFrame): Gradient evaluations
        prefix (str): Column name prefix; default="D"
        outvar (str): Name to give output id column; default="output"
        lambvar (str): Name to give eigenvalue column; default="lam"

    Returns:
        DataFrame: Active subspace directions and eigenvalues

    References:
        Constantine, "Active Subspaces" (2015) SIAM

    Examples:

        >>> import grama as gr
        >>> from grama.models import make_cantilever_beam
        >>> md = make_cantilever_beam()
        >>> df_base = md >> gr.ev_monte_carlo(n=1e2, df_det="nom", skip=True)
        >>> df_grad = md >> gr.ev_grad_fd(df_base=df_base)
        >>> df_as = df_grad >> gr.tf_asub()

    """
    ## Setup
    res = list(map(lambda s: s.split("_" + prefix, 1), df.columns))
    all_outputs = list(map(lambda r: re.sub("^" + prefix, "", r[0]), res))
    all_inputs = list(map(lambda r: r[1], res))
    outputs = list(set(all_outputs))

    list_df = []

    ## Loop
    for output in outputs:
        bool_test = list(map(lambda s: s == output, all_outputs))
        U, s, Vh = svd(df.loc[:, bool_test].values)

        df_tmp = DataFrame(
            data=Vh, columns=list(itertools.compress(all_inputs, bool_test))
        )
        df_tmp[lamvar] = s
        df_tmp[outvar] = [output] * len(s)
        list_df.append(df_tmp)

    return concat(list_df).reset_index(drop=True)


@pipe
def tf_asub(*args, **kwargs):
    return tran_asub(*args, **kwargs)


# --------------------------------------------------
## Inner product
@curry
def tran_inner(df, df_weights, prefix="dot", name=None, append=True):
    r"""Inner products

    Compute inner product of target df with weights defined by df_weights.

    Args:
        df (DataFrame): Data to compute inner products against
        df_weights (DataFrame): Weights for inner prodcuts
        prefix (str): Name prefix for resulting inner product columns;
            default="dot"
        name (str): Name of identity column in df_weights or None
        append (bool): Append new data to original DataFrame?

    Returns:
        DataFrame: Results of inner products

    Examples:

        >>> ## Setup
        >>> from dfply import *
        >>> import grama as gr
        >>> from grama.models import make_cantilever_beam
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>> md = make_cantilever_beam()
        >>> # Generate active subspace results
        >>> df_base = md >> gr.ev_monte_carlo(n=1e2, df_det="nom")
        >>> df_grad = md >> gr.ev_grad_fd(df_base=df_base)
        >>> df_as = df_grad >> \
        >>>     gr.tf_asub() >> \
        >>>     group_by(X.out) >> \
        >>>     mask(min_rank(-X.lam) == 1) >> \
        >>>     ungroup()
        >>> # Post-process
        >>> df_reduce = gr.tran_inner(df_base, df_as, name="out")
        >>> sns.scatterplot(
        >>>     data=df_reduce,
        >>>     x="dot_g_stress",
        >>>     y="g_stress"
        >>> )
        >>> plt.show()
        >>> sns.scatterplot(
        >>>     data=df_reduce,
        >>>     x="dot_g_disp",
        >>>     y="g_disp"
        >>> )
        >>> plt.show()

    """
    ## Check invariants
    if df_weights.shape[0] == 0:
        raise ValueError("df_weights cannot be empty")
    if isinstance(name, str):
        if not (name in df_weights.columns):
            raise ValueError("name must be column of df_weights or None")

    ## Check column overlap
    diff = set(df_weights.columns).difference(set(df.columns))
    if len(diff) > 0:
        warnings.warn("ignoring df_weights columns {}".format(diff), UserWarning)
    comm = list(set(df_weights.columns).difference(diff))

    ## Compute inner products
    dot_prod = dot(df[comm].values, df_weights[comm].values.T)

    ## Construct output dataframe
    if df_weights.shape[0] == 1:
        df_res = DataFrame(data={prefix: dot_prod.flatten()})

    elif df_weights.shape[0] > 1:
        if name is None:
            names = list(map(lambda i: prefix + str(i), range(dot_prod.shape[1])))
        else:
            names = list(
                map(
                    lambda i: prefix + "_" + df_weights[name].values[i],
                    range(dot_prod.shape[1]),
                )
            )

        df_res = DataFrame(data=dot_prod, columns=names)

    if append:
        df_res = concat((df.reset_index(drop=True), df_res), axis=1)

    return df_res


@pipe
def tf_inner(*args, **kwargs):
    return tran_inner(*args, **kwargs)


# --------------------------------------------------
## Describe
@curry
def tran_describe(df):
    """Describe a dataframe

    Synonym for Pandas df.describe()

    Args:
        df (DataFrame): Data to describe

    Returns:
        Printed summary

    """
    return df.describe()


@pipe
def tf_describe(*args, **kwargs):
    return tran_describe(*args, **kwargs)
