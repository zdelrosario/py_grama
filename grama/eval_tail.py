__all__ = [
    "eval_form_pma",
    "ev_form_pma",
    "eval_form_ria",
    "ev_form_ria",
]

import grama as gr
from grama import pipe, custom_formatwarning
from numpy import array, argmin, ones, eye, zeros, sqrt, NaN
from numpy.linalg import norm as length
from numpy.random import multivariate_normal
from pandas import DataFrame, concat
from scipy.optimize import minimize
from toolz import curry

## Utility Functions
# --------------------------------------------------
def make_T(model, df_corr):
    """Make covariance matrix from list

    Construct a covariance matrix from given entries. Check for correctness
    against a model.

    """
    ## TODO
    raise NotImplementedError


## FORM
# --------------------------------------------------
@curry
def eval_form_pma(
    model,
    betas=None,
    cons=None,
    df_corr=None,
    df_det=None,
    append=True,
    tol=1e-3,
    maxiter=25,
    nrestart=1,
):
    r"""Tail quantile via FORM PMA

    Approximate the desired tail quantiles using the performance measure
    approach (PMA) of the first-order reliability method (FORM) [1]. Select
    limit states to minimize at desired quantile with `betas`. Provide
    confidence levels `cons` and estimator covariance `df_corr` to compute with
    margin in beta [2].

    Args:
        model (gr.Model): Model to analyze
        betas (dict): Target reliability indices;
            key   = limit state name; must be in model.out
            value = reliability index; beta = Phi^{-1}(reliability)
        cons (dict or None): Target confidence levels;
            key   = limit state name; must be in model.out
            value = confidence level, \in (0, 1)
        df_corr (DataFrame or None): Sampling distribution covariance entries;
            parameters with no information assumed to be known exactly.
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        append (bool): Append MPP results for random values?

    Returns:
        DataFrame: Results of MPP search

    Notes:
        - Since FORM PMA relies on optimization over the limit state, it is
          often beneficial to scale your limit state to keep values near unity.

    References:
        - [1] Tu, Choi, and Park, "A new study on reliability-based design optimization," Journal of Mechanical Design, 1999
        - [2] del Rosario, Fenrich, and Iaccarino, "Fast precision margin with the first-order reliability method," AIAA Journal, 2019

    """
    ## Check invariants
    if betas is None:
        raise ValueError(
            "Must provide `betas` keyword argument to define reliability targets"
        )
    if not set(betas.keys()).issubset(set(model.out)):
        raise ValueError("betas.keys() must be subset of model.out")
    if not cons is None:
        if not (set(cons.keys()) == set(betas.keys())):
            raise ValueError("cons.keys() must be same as betas.keys()")
        else:
            if df_corr is None:
                raise ValueError("Must provide df_corr is using cons")
        raise NotImplementedError

    df_det = model.var_outer(
        DataFrame(data=zeros((1, model.n_var_rand)), columns=model.var_rand),
        df_det=df_det,
    )
    df_det = df_det[model.var_det]

    # df_return = DataFrame(columns=model.var_rand + model.var_det + list(betas.keys()))
    df_return = DataFrame()
    for ind in range(df_det.shape[0]):
        ## Loop over objectives
        for key in betas.keys():
            ## Temp dataframe
            df_inner = df_det.iloc[[ind]].reset_index(drop=True)

            ## Construct lambdas
            def objective(z):
                ## Transform: standard normal-to-random variable
                df_norm = DataFrame(data=[z], columns=model.var_rand)
                df_rand = model.norm2rand(df_norm)
                df = model.var_outer(df_rand, df_det=df_inner)

                df_res = gr.eval_df(model, df=df)
                g = df_res[key].iloc[0]

                # return (g, jac)
                return g

            def con_beta(z):
                return z.dot(z) - (betas[key]) ** 2

            ## Use conservative direction for initial guess
            signs = array([
                model.density.marginals[k].sign for k in model.var_rand
            ])
            if length(signs) > 0:
                z0 = betas[key] * signs / length(signs)
            else:
                z0 = betas[key] * ones(model.n_var_rand) / sqrt(model.n_var_rand)

            ## Minimize
            res_all = []
            for jnd in range(nrestart):
                res = minimize(
                    objective,
                    z0,
                    args=(),
                    method="SLSQP",
                    jac=False,
                    tol=tol,
                    options={"maxiter": maxiter, "disp": False},
                    constraints=[{"type": "eq", "fun": con_beta}],
                )
                # Append only a successful result
                if res["status"] == 0:
                    res_all.append(res)
                # Set a random start; repeat
                z0 = multivariate_normal([0] * model.n_var_rand, eye(model.n_var_rand))
                z0 = z0 / length(z0) * betas[key]

            # Choose value among restarts
            if len(res_all) > 0:
                i_star = argmin([res.fun for res in res_all])
                x_star = res_all[i_star].x
                fun_star = res_all[i_star].fun
            else:
                ## WARNING
                x_star = [NaN] * model.n_var_rand
                fun_star = NaN

            ## Extract results
            if append:
                df_inner = concat(
                    (df_inner, DataFrame(data=[x_star], columns=model.var_rand)),
                    axis=1,
                    sort=False,
                )
            df_inner[key] = [fun_star]
            df_return = concat((df_return, df_inner), axis=0, sort=False)

    return df_return


@pipe
def ev_form_pma(*args, **kwargs):
    return eval_form_pma(*args, **kwargs)


@curry
def eval_form_ria(
    model,
    limits=None,
    cons=None,
    df_corr=None,
    df_det=None,
    append=True,
    tol=1e-3,
    maxiter=25,
    nrestart=1,
):
    r"""Tail reliability via FORM RIA

    Approximate the desired tail probability using the reliability index
    approach (RIA) of the first-order reliability method (FORM) [1]. Select
    limit states to analyze with list input `limits`. Provide confidence levels
    `cons` and estimator covariance `df_corr` to compute with margin in beta
    [2].

    Args:
        model (gr.Model): Model to analyze
        limits (list): Target limit states; must be in model.out; limit state
            assumed to be critical at g == 0
        cons (dict or None): Target confidence levels;
            key   = limit state name; must be in model.out
            value = confidence level, \in (0, 1)
        df_corr (DataFrame or None): Sampling distribution covariance entries;
            parameters with no information assumed to be known exactly.
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        append (bool): Append MPP results for random values?

    Returns:
        DataFrame: Results of MPP search

    Notes:
        - Since FORM RIA relies on optimization over the limit state, it is
          often beneficial to scale your limit state to keep values near unity.

    References:
        - [1] Tu, Choi, and Park, "A new study on reliability-based design optimization," Journal of Mechanical Design, 1999
        - [2] del Rosario, Fenrich, and Iaccarino, "Fast precision margin with the first-order reliability method," AIAA Journal, 2019

    """
    ## Check invariants
    if limits is None:
        raise ValueError(
            "Must provide `limits` keyword argument to define reliability targets"
        )
    if not set(limits).issubset(set(model.out)):
        raise ValueError("`limits` must be subset of model.out")
    if not cons is None:
        if not (set(cons.keys()) == set(limits)):
            raise ValueError("cons.keys() must be same as limits")
        else:
            if df_corr is None:
                raise ValueError("Must provide df_corr is using cons")
        raise NotImplementedError

    df_det = model.var_outer(
        DataFrame(data=zeros((1, model.n_var_rand)), columns=model.var_rand),
        df_det=df_det,
    )
    df_det = df_det[model.var_det]

    # df_return = DataFrame(columns=model.var_rand + model.var_det + limits)
    df_return = DataFrame()
    for ind in range(df_det.shape[0]):
        ## Loop over objectives
        for key in limits:
            ## Temp dataframe
            df_inner = df_det.iloc[[ind]].reset_index(drop=True)

            ## Construct lambdas
            def fun_jac(z):
                ## Squared reliability index
                fun = z.dot(z)
                jac = 2 * z * length(z)

                return (fun, jac)

            def con_limit(z):
                ## Transform: standard normal-to-random variable
                df_norm = DataFrame(data=[z], columns=model.var_rand)
                df_rand = model.norm2rand(df_norm)
                df = model.var_outer(df_rand, df_det=df_inner)

                ## Eval limit state
                df_res = gr.eval_df(model, df=df)
                g = df_res[key].iloc[0]

                return g

            ## Use conservative direction for initial guess
            signs = array([
                model.density.marginals[k].sign for k in model.var_rand
            ])
            if length(signs) > 0:
                z0 = signs / length(signs)
            else:
                z0 = ones(model.n_var_rand) / sqrt(model.n_var_rand)

            ## Minimize
            res_all = []
            for jnd in range(nrestart):
                res = minimize(
                    fun_jac,
                    z0,
                    args=(),
                    method="SLSQP",
                    jac=True,
                    tol=tol,
                    options={"maxiter": maxiter, "disp": False},
                    constraints=[{"type": "eq", "fun": con_limit}],
                )
                # Append only a successful result
                if res["status"] == 0:
                    res_all.append(res)
                # Set a random start; repeat
                z0 = multivariate_normal([0] * model.n_var_rand, eye(model.n_var_rand))
                z0 = z0 / length(z0)

            # Choose value among restarts
            if len(res_all) > 0:
                i_star = argmin([res.fun for res in res_all])
                x_star = res_all[i_star].x
                fun_star = sqrt(res_all[i_star].fun)
            else:
                ## WARNING
                x_star = [NaN] * model.n_var_rand
                fun_star = NaN

            ## Extract results
            if append:
                df_inner = concat(
                    (df_inner, DataFrame(data=[x_star], columns=model.var_rand)),
                    axis=1,
                    sort=False,
                )
            df_inner[key] = [fun_star]
            df_return = concat((df_return, df_inner), axis=0, sort=False)

    return df_return


@pipe
def ev_form_ria(*args, **kwargs):
    return eval_form_ria(*args, **kwargs)
