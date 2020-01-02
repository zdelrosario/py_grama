__all__ = [
    "tran_bootstrap",
    "tf_bootstrap",
    "tran_outer",
    "tf_outer",
    "tran_gather",
    "tf_gather",
    "tran_spread",
    "tf_spread"
]

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from grama import pipe
from toolz import curry
from numbers import Integral

## Bootstrap utility
@curry
def tran_bootstrap(
        df,
        tran=None,
        n_boot=1000,
        n_sub=25,
        con=0.99,
        col_sel=None,
        seed=None
):
    """Estimate bootstrap confidence intervals

    Estimate bootstrap confidence intervals for a given transform. Uses the
    "bootstrap-t" procedure discussed in Efron and Tibshirani (1993).

    Args:
        df (DataFrame): Data to bootstrap
        tran (grama tran_ function): Transform procedure which generates statistic
        n_boot (numeric): Monte Carlo resamples for bootstrap, default = 1000
        n_sub (numeric): Nested resamples to estimate SE, default = 25
        con (float): Confidence level, default = 0.99
        col_sel (list(string)): Columns to include in bootstrap calculation

    Returns:
        DataFrame: Results of tran(df), plus _lo and _hi columns for
        numeric columns

    References and notes:
        Efron and Tibshirani (1993)

      "The bootstrap-t procedure... is particularly applicable to location
       statistics like the sample mean.... The bootstrap-t method, at least in
       its simple form, cannot be trusted for more general problems, like
       setting a confidence interval for a correlation coefficient."

    Examples:

    """
    ## Set seed only if given
    if seed is not None:
        np.random.seed(seed)

    ## Ensure sample count is int
    if not isinstance(n_boot, Integral):
        print("tran_bootstrap() is rounding n_boot...")
        n_boot = int(n_boot)
    if not isinstance(n_sub, Integral):
        print("tran_bootstrap() is rounding n_sub...")
        n_sub = int(n_sub)

    ## Base results
    df_base = tran(df)

    ## Select columns for bootstrap
    col_numeric = list(
        df_base.select_dtypes(include="number")
               .columns
    )
    if not (col_sel is None):
        col_numeric = list(
            set(col_numeric).intersection(set(col_sel))
        )

    ## Setup
    n_samples = df.shape[0]
    n_row = df_base.shape[0]
    n_col = len(col_numeric)
    alpha = (1 - con) / 2
    theta_hat = df_base[col_numeric].values

    theta_all   = np.zeros((n_boot, n_row, n_col))
    se_boot_all = np.zeros((n_boot, n_row, n_col))
    z_all       = np.zeros((n_boot, n_row, n_col))
    theta_sub   = np.zeros((n_sub,  n_row, n_col))

    ## Main loop
    for ind in range(n_boot):
        ## Construct resample
        Ib = np.random.choice(n_samples, size=n_samples, replace=True)
        theta_all[ind] = tran(df.iloc[Ib, ])[col_numeric].values

        ## Internal loop to approximate SE
        for jnd in range(n_sub):
            Isub = Ib[np.random.choice(n_samples, size=n_samples, replace=True)]
            theta_sub[jnd] = tran(df.iloc[Isub, ])[col_numeric].values
        se_boot_all[ind] = np.std(theta_sub, axis=0)

        ## Construct approximate pivot
        z_all[ind] = (theta_all[ind] - theta_hat) / se_boot_all[ind]

    ## Compute bootstrap table
    t_lo, t_hi = np.quantile(z_all, q=[1 - alpha, alpha], axis=0)

    ## Estimate bootstrap intervals
    se = np.std(theta_all, axis=0)
    theta_lo = theta_hat - t_lo * se
    theta_hi = theta_hat - t_hi * se

    ## Assemble output data
    col_lo = list(map(
        lambda s: s + "_lo",
        col_numeric
    ))
    col_hi = list(map(
        lambda s: s + "_hi",
        col_numeric
    ))

    df_lo = pd.DataFrame(data=theta_lo, columns=col_lo)
    df_hi = pd.DataFrame(data=theta_hi, columns=col_hi)

    df_lo.index = df_base.index
    df_hi.index = df_base.index

    return pd.concat((df_base, df_lo, df_hi), axis=1)

@pipe
def tf_bootstrap(*args, **kwargs):
    return tran_bootstrap(*args, **kwargs)

## DataFrame outer product
@curry
def tran_outer(df, df_outer):
    """Outer merge

    Perform an outer-merge on two dataframes.

    Args:
        df (DataFrame): Data to merge
        df_outer (DataFrame): Data to merge; outer

    Returns:
        DataFrame: Merged data

    Examples:
        >>> import grama as gr
        >>> import pandas as pd
        >>> df = pd.DataFrame(dict(x=[1,2]))
        >>> df_outer = pd.DataFrame(dict(y=[3,4]))
        >>> df_res = gr.tran_outer(df, df_outer)
        >>> df_res
        >>>    x  y
        >>> 0  1  3
        >>> 1  2  3
        >>> 2  1  4
        >>> 3  2  4
    """
    n_rows = df.shape[0]
    list_df = []

    for ind in range(df_outer.shape[0]):
        df_rep = pd.concat([df_outer.loc[[ind]]] * n_rows, ignore_index=True)
        list_df.append(pd.concat((df, df_rep), axis=1))

    return pd.concat(list_df, ignore_index=True)

@pipe
def tf_outer(*args, **kwargs):
    return tran_outer(*args, **kwargs)

## Reshape functions
# --------------------------------------------------
@curry
def tran_gather(df, key, value, cols):
    """Makes a DataFrame longer by gathering columns.

    """
    id_vars = [col for col in df.columns if col not in cols]
    id_values = cols
    var_name = key
    value_name = value

    return pd.melt(
        df,
        id_vars,
        id_values,
        var_name=var_name,
        value_name=value_name
    )

@pipe
def tf_gather(*args, **kwargs):
    return tran_gather(*args, **kwargs)

@curry
def tran_spread(df, key, value, fill=np.nan, drop=False):
    """Makes a DataFrame wider by spreading columns.

    """
    index = [col for col in df.columns if ((col != key) and (col != value))]

    df_new = df.pivot_table(
        index=index,
        columns=key,
        values=value,
        fill_value=fill
    ).reset_index()

    ## Drop extraneous info
    df_new = df_new.rename_axis(None, axis=1)
    if drop:
        df_new.drop("index", axis=1, inplace=True)

    return df_new

@pipe
def tf_spread(*args, **kwargs):
    return tran_spread(*args, **kwargs)
