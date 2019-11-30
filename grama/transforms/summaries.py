__all__ = [
    "tran_sobol",
    "tf_sobol"
]

import numpy as np
import pandas as pd

from .. import core
from ..core import pipe
from toolz import curry

## Compute Sobol' indices
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
