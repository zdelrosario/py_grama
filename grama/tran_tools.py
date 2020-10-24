__all__ = [
    "tran_angles",
    "tf_angles",
    "tran_bootstrap",
    "tf_bootstrap",
    "tran_copula_corr",
    "tf_copula_corr",
    "tran_outer",
    "tf_outer",
    "tran_kfolds",
    "tf_kfolds",
]

from collections import ChainMap
from numpy import arange, ceil, zeros, std, quantile, nan, triu_indices, unique
from numpy.random import choice, permutation
from numpy.random import seed as set_seed
from pandas import concat, DataFrame, melt

from grama import add_pipe, pipe, copy_meta, Intention, mse, rsq
from grama import (
    tf_bind_cols,
    tf_filter,
    tf_summarize,
    tf_drop,
    tf_mutate,
    var_in,
    ev_df,
)

from toolz import curry
from numbers import Integral
from pandas.api.types import is_numeric_dtype
from scipy.linalg import subspace_angles
from scipy.stats import norm

X = Intention()

## k-Fold CV utility
# --------------------------------------------------
@curry
def tran_kfolds(
    df,
    k=None,
    ft=None,
    var_fold=None,
    summaries=None,
    tf=tf_summarize,
    shuffle=True,
    seed=None,
):
    r"""Perform k-fold CV

    Perform k-fold cross-validation (CV) using a given fitting procedure (ft).
    Optionally provide

    Args:
        df (DataFrame): Data to pass to given fitting procedure
        ft (gr.ft_): Partially-evaluated grama fit function; defines model fitting
            procedure and outputs to aggregate
        tf (gr.tf_): Partially-evaluated grama transform function; evaluation of
            fitted model will be passed to tf and provided with keyword arguments
            from summaries
        var_fold (str or None):
        summaries (dict of functions): Summary functions to pass to tf; will be evaluated
            for each output of ft. Each summary must have signature summary(f_pred, f_meas)
        k (int): Number of folds; k=5 to k=10 recommended [1]
        shuffle (bool): Shuffle the data before CV?

    Notes:
        - Many grama functions support /partial evaluation/; this allows one to specify things like hyperparameters in fitting functions without providing data and executing the fit. You can take advantage of this functionality to easly do hyperparameter studies.

    Returns:
        DataFram: Aggregated results within each of k-folds using given model and
            summary transform

    References:
        James, Witten, Hastie, and Tibshirani, "An introduction to statistical learning" (2017), Chapter 5. Resampling Methods

    Examples:

        >>> import grama as gr
        >>> from grama.data import df_stang
        >>> from grama.fit import ft_rf
        >>> df_kfolds = (
        >>>     df_stang
        >>>     >> gr.tf_kfolds(
        >>>         k=5,
        >>>         ft=ft_rf(out=["thick"], var=["E", "mu"]),
        >>>     )

    """
    ## Check invariants
    if ft is None:
        raise ValueError("Must provide ft keyword argument")
    if (k is None) and (var_fold is None):
        print("... tran_kfolds is using default k=5")
        k = 5
    if summaries is None:
        print("... tran_kfolds is using default summaries mse and rsq")
        summaries = dict(mse=mse, rsq=rsq)

    n = df.shape[0]
    ## Handle custom folds
    if not (var_fold is None):
        ## Check for a valid var_fold
        if not (var_fold in df.columns):
            raise ValueError("var_fold must be in df.columns or None")
        ## Build folds
        levels = unique(df[var_fold])
        k = len(levels)
        print("... tran_kfolds found {} levels via var_folds".format(k))
        Is = []
        for l in levels:
            Is.append(list(arange(n)[df[var_fold] == l]))

    else:
        ## Shuffle data indices
        if shuffle:
            if seed:
                set_seed(seed)
            I = permutation(n)
        else:
            I = arange(n)
        ## Build folds
        di = int(ceil(n / k))
        Is = [I[i * di : min((i + 1) * di, n)] for i in range(k)]

    ## Iterate over folds
    df_res = DataFrame()
    for i in range(k):
        ## Train by out-of-fold data
        md_fit = df >> tf_filter(~var_in(X.index, Is[i])) >> ft

        ## Test by in-fold data
        df_pred = md_fit >> ev_df(
            df=df >> tf_filter(var_in(X.index, Is[i])) >> tf_drop(md_fit.out),
            append=False,
        )

        # Modify names with suffix
        out_pred = list(map(lambda s: s + "_pred", md_fit.out))
        df_pred.rename(mapper=dict(zip(md_fit.out, out_pred)), axis=1, inplace=True)

        ## Specialize summaries for output names
        summaries_all = ChainMap(
            *[
                {
                    key + "_" + out: fun(X[out + "_pred"], X[out])
                    for key, fun in summaries.items()
                }
                for out in md_fit.out
            ]
        )

        ## Aggregate
        df_summary_tmp = (
            df_pred
            >> tf_bind_cols(df[md_fit.out] >> tf_filter(var_in(X.index, Is[i])))
            >> tf(**summaries_all)
            # >> tf_mutate(_kfold=i)
        )

        if var_fold is None:
            df_summary_tmp = df_summary_tmp >> tf_mutate(_kfold=i)
        else:
            df_summary_tmp[var_fold] = levels[i]

        df_res = concat((df_res, df_summary_tmp), axis=0).reset_index(drop=True)

    return df_res


tf_kfolds = add_pipe(tran_kfolds)

## Bootstrap utility
# --------------------------------------------------
@curry
def tran_bootstrap(
    df, tran=None, n_boot=500, n_sub=25, con=0.90, col_sel=None, seed=None
):
    r"""Estimate bootstrap confidence intervals

    Estimate bootstrap confidence intervals for a given transform. Uses the
    "bootstrap-t" procedure discussed in Efron and Tibshirani (1993).

    Args:
        df (DataFrame): Data to bootstrap
        tran (grama tran_ function): Transform procedure which generates statistic
        n_boot (numeric): Monte Carlo resamples for bootstrap
        n_sub (numeric): Nested resamples to estimate SE
        con (float): Confidence level
        col_sel (list(string)): Columns to include in bootstrap calculation

    Returns:
        DataFrame: Results of tran(df), plus _lo and _up columns for
        numeric columns

    References and notes:
       Efron and Tibshirani (1993) "The bootstrap-t procedure... is
       particularly applicable to location statistics like the sample mean....
       The bootstrap-t method, at least in its simple form, cannot be trusted
       for more general problems, like setting a confidence interval for a
       correlation coefficient."

    Examples:

    """
    ## Set seed only if given
    if seed is not None:
        set_seed(seed)

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
    col_numeric = list(df_base.select_dtypes(include="number").columns)
    if not (col_sel is None):
        col_numeric = list(set(col_numeric).intersection(set(col_sel)))

    ## Setup
    n_samples = df.shape[0]
    n_row = df_base.shape[0]
    n_col = len(col_numeric)
    alpha = (1 - con) / 2
    theta_hat = df_base[col_numeric].values

    theta_all = zeros((n_boot, n_row, n_col))
    se_boot_all = zeros((n_boot, n_row, n_col))
    z_all = zeros((n_boot, n_row, n_col))
    theta_sub = zeros((n_sub, n_row, n_col))

    ## Main loop
    for ind in range(n_boot):
        ## Construct resample
        Ib = choice(n_samples, size=n_samples, replace=True)
        df_tmp = copy_meta(df, df.iloc[Ib,])
        theta_all[ind] = tran(df_tmp)[col_numeric].values

        ## Internal loop to approximate SE
        for jnd in range(n_sub):
            Isub = Ib[choice(n_samples, size=n_samples, replace=True)]
            df_tmp = copy_meta(df, df.iloc[Isub,])
            theta_sub[jnd] = tran(df_tmp)[col_numeric].values
        se_boot_all[ind] = std(theta_sub, axis=0)

        ## Construct approximate pivot
        z_all[ind] = (theta_all[ind] - theta_hat) / se_boot_all[ind]

    ## Compute bootstrap table
    t_lo, t_hi = quantile(z_all, q=[1 - alpha, alpha], axis=0)

    ## Estimate bootstrap intervals
    se = std(theta_all, axis=0)
    theta_lo = theta_hat - t_lo * se
    theta_hi = theta_hat - t_hi * se

    ## Assemble output data
    col_lo = list(map(lambda s: s + "_lo", col_numeric))
    col_hi = list(map(lambda s: s + "_up", col_numeric))

    df_lo = DataFrame(data=theta_lo, columns=col_lo)
    df_hi = DataFrame(data=theta_hi, columns=col_hi)

    df_ci = concat((df_lo, df_hi), axis=1).sort_index(axis=1)
    df_ci.index = df_base.index

    return concat((df_base, df_ci), axis=1)


tf_bootstrap = add_pipe(tran_bootstrap)


## DataFrame outer product
# --------------------------------------------------
@curry
def tran_outer(df, df_outer):
    r"""Outer merge

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
        df_rep = concat([df_outer.iloc[[ind]]] * n_rows, ignore_index=True)
        list_df.append(concat((df, df_rep), axis=1))

    return concat(list_df, ignore_index=True)


tf_outer = add_pipe(tran_outer)


## Assess subspace angles
# --------------------------------------------------
def tran_angles(df, df2):
    r"""Subspace angles

    Compute the subspace angles between two matrices. A wrapper for
    scipy.linalg.subspace_angles that corrects for column ordering. Row ordering
    is assumed.

    Args:
        df (DataFrame): First matrix to compare
        df2 (DataFrame): Second matrix to compare

    Returns:
        array: Array of angles (in radians)

    Examples:

        >>> import grama as gr
        >>> import pandas as pd
        >>> df = pd.DataFrame(dict(v=[+1, +1]))
        >>> df_v1 = pd.DataFrame(dict(w=[+1, -1]))
        >>> df_v2 = pd.DataFrame(dict(w=[+1, +1]))
        >>> theta1 = angles(df, df_v1)
        >>> theta2 = angles(df, df_v2)

    """
    ## Compute subspace angles
    A1 = df.values
    A2 = df2.values

    return subspace_angles(A1, A2)


tf_angles = add_pipe(tran_angles)


## Compute Gaussian copula correlations from data
# --------------------------------------------------
def tran_copula_corr(df, model=None, density=None):
    r"""Compute Gaussian copula correlations from data

    Convenience function to fit a Gaussian copula (correlations) based on data
    and pre-fitted marginals. Intended for use with gr.comp_copula_gaussian().
    Must provide either `model` or `density`.

    Args:
        df (DataFrame): Matrix of data for correlation estimation
        model (gr.Model): Model with defined marginals
        density (gr.Density): Density with defined marginals

    Returns:
        DataFrame: Correlation data ready for use with gr.comp_copula_gaussian()

    Examples:

        >>> import grama as gr
        >>> from grama.data import df_stang
        >>> md = gr.Model() >> \
        >>>     gr.cp_marginals(
        >>>         E=gr.marg_named(df_stang.E, "norm"),
        >>>         mu=gr.marg_named(df_stang.mu, "beta"),
        >>>         thick=gr.marg_named(df_stang.thick, "norm")
        >>>     )
        >>> df_corr = gr.tran_copula_corr(df_stang, model=md)

    """
    if density is None:
        density = model.density

    ## Check invariants
    if not set(density.marginals.keys()).issubset(set(df.columns)):
        raise ValueError("df must have columns for all var_rand")

    ## Convert data
    df_res = density.sample2pr(df)
    df_norm = df_res.apply(norm.ppf)

    ## Compute correlations
    df_mat = df_norm.corr()
    Ind = triu_indices(len(density.marginals), 1)

    ## Arrange
    var_rand = df_mat.columns
    var1_all = []
    var2_all = []
    corr_all = []

    for i, j in zip(Ind[0], Ind[1]):
        var1_all.append(var_rand[i])
        var2_all.append(var_rand[j])
        corr_all.append(df_mat.iloc[i, j])

    return DataFrame(dict(var1=var1_all, var2=var2_all, corr=corr_all))


tf_copula_corr = add_pipe(tran_copula_corr)
