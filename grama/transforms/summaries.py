__all__ = [
    "tran_sobol",
    "tf_sobol",
    "tran_directions",
    "tf_directions",
    "tran_inner",
    "tf_inner"
]

import numpy as np
import pandas as pd
import re
import itertools
import warnings

from ..tools import pipe
from toolz import curry

## Compute Sobol' indices
# --------------------------------------------------
@curry
def tran_sobol(
        df,
        varname="hybrid_var",
        typename="var",
        plan="first"
):
    """Estimate Sobol' indices based on hybrid point evaluations
    """
    if not (varname in df.columns):
        raise ValueError("{} not in df.columns".format(varname))

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
            df_tau[typename] = v_model[i_var]

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
            df_tau[typename] = v_model[i_var]

            df_index = df_tau.drop(columns=typename) \
                             .reset_index(drop=True) \
                             .truediv(df_var.drop(columns=typename))
            df_index[typename] = "T_" + v_model[i_var]

            df_res = pd.concat((df_res, df_tau, df_index))
    else:
        raise ValueError("plan `{}` not valid".format(plan))

    return df_res

@pipe
def tf_sobol(*args, **kwargs):
    return tran_sobol(*args, **kwargs)

## Linear algebra tools
# --------------------------------------------------
## Gradient principal directions (AS)
@curry
def tran_directions(df, prefix="D", outvar="output", lamvar="lam"):
    """Compute principal directions and eigenvalues for all outputs
    based on output of ev_grad_fd()
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
def tf_directions(*args, **kwargs):
    return tran_directions(*args, **kwargs)

## Inner product
@curry
def tran_inner(df, df_weights, prefix="dot", append=False):
    """Compute inner products

    @param df DataFrame data to compute inner products against
    @param df_weights DataFrame weights for inner prodcuts
    @prefix string name prefix for resulting inner product columns
    @append bool append new data to original DataFrame?
    """
    ## Check invariants
    if df_weights.shape[0] == 0:
        raise ValueError("df_weights cannot be empty!")

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
        df_res = pd.DataFrame(
            data = dot,
            columns = list(map(
                lambda i: prefix + str(i),
                range(dot.shape[1])
            ))
        )

    if append:
        df_res = df.join(df_res)

    return df_res


@pipe
def tf_inner(*args, **kwargs):
    return tran_inner(*args, **kwargs)
