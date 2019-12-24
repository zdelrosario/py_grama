__all__ = [
    "tran_sobol",
    "tf_sobol",
    "tran_asub",
    "tf_asub",
    "tran_inner",
    "tf_inner"
]

import numpy as np
import pandas as pd
import re
import itertools
import warnings

from ..tools import pipe, custom_formatwarning
from toolz import curry

warnings.formatwarning = custom_formatwarning

## Compute Sobol' indices
# --------------------------------------------------
@curry
def tran_sobol(
        df,
        varname="hybrid_var",
        typename="ind",
        digits=2
):
    """Post-process results from gr.eval_hybrid()

    Estimate Sobol' indices based on hybrid point evaluations (Sobol', 1999).
    Intended as post-processor for gr.eval_hybrid().

    Args:
        df (DataFrame): Hybrid point results from gr.eval_hybrid()
        varname (str): Column name to give for sweep variable; default="hybrid_var"
        typename (str): Name to give index type column in results
        digits (int): Number of digits for rounding reported results

    Returns:
        DataFrame: Sobol' indices

    Notes:
        - Index type ["first", "total"] is inferred from input df._meta;
          this is assigned by gr.eval_hybrid().

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
    if not (varname in df.columns):
        raise ValueError("{} not in df.columns".format(varname))

    ## Determine plan from dataframe metadata
    metadata = df._meta
    if "ev_hybrid" in metadata:
        plan = metadata[10:]
    else:
        raise ValueError("df not hybrid points!")

    ## Setup
    v_model = list(filter(
        lambda s: s != "_",
        df[[varname]].drop_duplicates()[varname]
    ))
    v_drop = v_model + [varname]
    n_in = len(v_model)
    I_base  = (df[varname] == "_")

    df_var = pd.DataFrame(df.drop(columns=v_drop).var()).transpose()
    df_var[typename] = "var"

    df_res = df_var.copy()

    if plan == "first":
        mu_base = df.drop(columns=v_drop)[I_base].mean()

        for i_var in range(n_in):
            I_var = (df[varname] == v_model[i_var])

            mu_var = df.drop(columns=v_drop)[I_var].mean()
            mu_tot = 0.5 * (mu_base + mu_var)
            s2_var = (
                df[I_base].drop(columns=v_drop).reset_index(drop=True).mul(
                    df[I_var].drop(columns=v_drop).reset_index(drop=True)
                )
            ).mean()

            df_tau = pd.DataFrame(s2_var - mu_tot**2).transpose()
            df_tau[typename] = "T_" + v_model[i_var]

            df_index = df_tau.drop(columns=typename) \
                             .reset_index(drop=True) \
                             .truediv(df_var.drop(columns=typename))
            df_index[typename] = "S_" + v_model[i_var]

            df_res = pd.concat((df_res, df_tau, df_index))

    elif plan == "total":
        for i_var in range(n_in):
            I_var = (df[varname] == v_model[i_var])
            s2_var = (
                (
                    df[I_base].drop(columns=v_drop).reset_index(drop=True) - \
                    df[I_var].drop(columns=v_drop).reset_index(drop=True)
                )**2
            ).mean() * 0.5

            df_tau = pd.DataFrame(s2_var).transpose()
            df_tau[typename] = "T_" + v_model[i_var]

            df_index = df_tau.drop(columns=typename) \
                             .reset_index(drop=True) \
                             .truediv(df_var.drop(columns=typename))
            df_index[typename] = "T_" + v_model[i_var]

            df_res = pd.concat((df_res, df_tau, df_index))
    else:
        raise ValueError("plan `{}` not valid".format(plan))

    ## Post-process
    outputs = df_res.drop(typename, axis=1).columns
    df_res[outputs] = df_res[outputs].apply(
        lambda row: np.round(row, decimals=digits)
    )
    df_res.sort_values(
        typename,
        inplace=True
    )

    return df_res

@pipe
def tf_sobol(*args, **kwargs):
    return tran_sobol(*args, **kwargs)

## Linear algebra tools
##################################################
## Gradient principal directions (AS)
@curry
def tran_asub(df, prefix="D", outvar="out", lamvar="lam"):
    """Active subspace estimator

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
        U, s, Vh = np.linalg.svd(df.loc[:, bool_test].values)

        df_tmp = pd.DataFrame(
            data=Vh,
            columns=list(itertools.compress(all_inputs, bool_test))
        )
        df_tmp[lamvar] = s
        df_tmp[outvar] = [output] * len(s)
        list_df.append(df_tmp)

    return pd.concat(list_df).reset_index(drop=True)

@pipe
def tf_asub(*args, **kwargs):
    return tran_asub(*args, **kwargs)

# --------------------------------------------------
## Inner product
@curry
def tran_inner(df, df_weights, prefix="dot", name=None, append=True):
    """Inner products

    Compute inner product of target df with weights defined by df_weights.

    Args:
        df (DataFrame): Data to compute inner products against
        df_weights (DataFrame): Weights for inner prodcuts
        prefix (str): Name prefix for resulting inner product columns;
            default="dot"
        name (str): Name of identity column in df_weights or None
        append (bool): Append new data to original DataFrame?; default=False

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
    dot = np.dot(df[comm].values, df_weights[comm].values.T)

    ## Construct output dataframe
    if df_weights.shape[0] == 1:
        df_res = pd.DataFrame(data = {prefix: dot.flatten()})

    elif df_weights.shape[0] > 1:
        if name is None:
            names = list(map(
                lambda i: prefix + str(i),
                range(dot.shape[1])
            ))
        else:
            names = list(map(
                lambda i: prefix + "_" + df_weights[name].values[i],
                range(dot.shape[1])
            ))

        df_res = pd.DataFrame(data=dot, columns=names)

    if append:
        df_res = df.join(df_res)

    return df_res

@pipe
def tf_inner(*args, **kwargs):
    return tran_inner(*args, **kwargs)
