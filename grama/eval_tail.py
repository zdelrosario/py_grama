__all__ = [
    "eval_form_pma",
    "ev_form_pma",
    "eval_form_ria",
    "ev_form_ria",
]

from grama import add_pipe, pipe, custom_formatwarning
from grama.eval_defaults import invariants_eval_model, invariants_eval_df
from .eval_defaults import eval_df
from numpy import array, argmin, ones, eye, zeros, sqrt, NaN, max
from numpy.linalg import norm as length
from numpy.random import multivariate_normal
from pandas import DataFrame, concat
from scipy.stats import norm
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
    rels=None,
    pofs=None,
    cons=None,
    df_corr=None,
    df_det=None,
    append=True,
    tol=1e-3,
    n_maxiter=25,
    n_restart=1,
    verbose=False,
):
    r"""Tail quantile via FORM PMA

    Approximate the desired tail quantiles using the performance measure approach (PMA) of the first-order reliability method (FORM) [1]. Select limit states to minimize at desired quantile with one of: `betas` (reliability indices), `rels` (reliability values), or `pofs` (probabilities of failure). Provide confidence levels `cons` and estimator covariance `df_corr` to compute with margin in beta [2].

    Note that under the performance measure approach, the optimized limit state value `g` is sought to be non-negative $g \geq 0$. This is usually included as a constraint in optimization, which can be accomplished in by using ``gr.eval_form_pnd()` *within* a model definition---see the Examples below for more details.

    Args:
        model (gr.Model): Model to analyze
        df_det (DataFrame): Deterministic levels for evaluation; use "nom" for nominal deterministic levels.

        betas (dict): Target reliability indices;
            key   = limit state name; must be in model.out
            value = reliability index; beta = Phi^{-1}(reliability)
        rels (dict): Target reliability values;
            key   = limit state name; must be in model.out
            value = reliability
        pofs (dict): Target probabilities of failure (POFs);
            key   = limit state name; must be in model.out
            value = probability of failure; pof = 1 - reliability

        cons (dict or None): Target confidence levels;
            key   = limit state name; must be in model.out
            value = confidence level, \in (0, 1)
        df_corr (DataFrame or None): Sampling distribution covariance entries; parameters with no information assumed to be known exactly.

        n_maxiter (int): Maximum iterations for each optimization run
        n_restart (int): Number of restarts (== number of optimization runs)
        append (bool): Append MPP results for random values?
        verbose (bool): Print optimization results?

    Returns:
        DataFrame: Results of MPP search

    Notes:
        Since FORM PMA relies on optimization over the limit state, it is often beneficial to scale your limit state to keep values near unity.

    References:
        Tu, Choi, and Park, "A new study on reliability-based design optimization," Journal of Mechanical Design, 1999
        del Rosario, Fenrich, and Iaccarino, "Fast precision margin with the first-order reliability method," AIAA Journal, 2019

    Examples::

        import grama as gr
        from grama.models import make_cantilever_beam
        md_beam = make_cantilever_beam()
        ## Evaluate the reliability of specified designs
        (
            md_beam
            >> gr.ev_form_pma(
                # Specify target reliability
                betas=dict(g_stress=3, g_disp=3),
                # Analyze three different thicknesses
                df_det=gr.df_make(t=[2, 3, 4], w=3)
            )
        )

        ## Specify reliability in POF form
        (
            md_beam
            >> gr.ev_form_pma(
                # Specify target reliability
                pofs=dict(g_stress=1e-3, g_disp=1e-3),
                # Analyze three different thicknesses
                df_det=gr.df_make(t=[2, 3, 4], w=3)
            )
        )

        ## Build a nested model for optimization under uncertainty
        md_opt = (
            gr.Model("Beam Optimization")
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(c_area=df.w * df.t),
                var=["w", "t"],
                out=["c_area"],
                name="Area objective",
            )
            >> gr.cp_vec_function(
                fun=lambda df: gr.eval_form_pma(
                    md_beam,
                    betas=dict(g_stress=3, g_disp=3),
                    df_det=df,
                    append=False,
                )
                var=["w", "t"],
                out=["g_stress", "g_disp"],
                name="Reliability constraints",
            )
            >> gr.cp_bounds(w=(2, 4), t=(2, 4))
        )
        # Run the optimization
        (
            md_opt
            >> gr.ev_min(
                out_min="c_area",
                out_geq=["g_stress", "g_disp"],
            )
        )

    """
    ## Check invariants
    invariants_eval_model(model)
    invariants_eval_df(df_corr, arg_name="df_corr", acc_none=True)
    invariants_eval_df(df_det, arg_name="df_det", valid_str=["nom"])
    # Check that reliability targets given by one argument only
    if all((betas is None, rels is None, pofs is None)):
        raise ValueError(
            "Must provide reliability targets as keyword argument `betas`, `rels`, OR `pofs`"
        )
    if sum((betas is not None, rels is not None, pofs is not None)) > 1:
        raise ValueError(
            "Must provide *only* one of `betas`, `rels`, OR `pofs`"
        )
    # Convert reliability targets to `betas`
    if rels is not None:
        betas = {k: norm.ppf(v) for (k, v) in rels.items()}
    if pofs is not None:
        betas = {k: norm.ppf(1 - v) for (k, v) in pofs.items()}

    # Check `betas` invariants
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

                df_res = eval_df(model, df=df)
                g = df_res[key].iloc[0]

                # return (g, jac)
                return g

            def con_beta(z):
                return z.dot(z) - (betas[key]) ** 2

            ## Use conservative direction for initial guess
            signs = array(
                [model.density.marginals[k].sign for k in model.var_rand]
            )
            if length(signs) > 0:
                z0 = betas[key] * signs / length(signs)
            else:
                z0 = (
                    betas[key] * ones(model.n_var_rand) / sqrt(model.n_var_rand)
                )

            ## Minimize
            res_all = []
            for jnd in range(n_restart):
                res = minimize(
                    objective,
                    z0,
                    args=(),
                    method="SLSQP",
                    jac=False,
                    tol=tol,
                    options={"maxiter": n_maxiter, "disp": False},
                    constraints=[{"type": "eq", "fun": con_beta}],
                )
                # Append only a successful result
                if res["status"] == 0:
                    res_all.append(res)
                # Set a random start; repeat
                z0 = multivariate_normal(
                    [0] * model.n_var_rand, eye(model.n_var_rand)
                )
                z0 = z0 / length(z0) * betas[key]

            # Choose value among restarts
            n_iter_total = sum([res_all[i].nit for i in range(len(res_all))])
            if len(res_all) > 0:
                i_star = argmin([res.fun for res in res_all])
                x_star = res_all[i_star].x
                fun_star = res_all[i_star].fun
                if verbose:
                    print("out = {}: Optimization successful".format(key))
                    print("n_iter = {}".format(res_all[i_star].nit))
                    print("n_iter_total = {}".format(n_iter_total))
            else:
                ## WARNING
                x_star = [NaN] * model.n_var_rand
                fun_star = NaN
                if verbose:
                    print("out = {}: Optimization unsuccessful".format(key))
                    print("n_iter = {}".format(res_all[i_star].nit))
                    print("n_iter_total = {}".format(n_iter_total))

            ## Extract results
            if append:
                df_inner = concat(
                    (
                        df_inner,
                        model.norm2rand(
                            DataFrame(data=[x_star], columns=model.var_rand)
                        ),
                    ),
                    axis=1,
                    sort=False,
                )
            df_inner[key] = [fun_star]
            df_return = concat((df_return, df_inner), axis=0, sort=False)

    if not append:
        df_return = (
            df_return.groupby(model.var_det)
            .agg({s: max for s in betas.keys()})
            .reset_index()
        )

    return df_return


ev_form_pma = add_pipe(eval_form_pma)


@curry
def eval_form_ria(
    model,
    limits=None,
    format="betas",
    cons=None,
    df_corr=None,
    df_det=None,
    append=True,
    tol=1e-3,
    n_maxiter=25,
    n_restart=1,
    verbose=False,
):
    r"""Tail reliability via FORM RIA

    Approximate the desired tail probability using the reliability index approach (RIA) of the first-order reliability method (FORM) [1]. Select limit states to analyze with list input `limits`. Choose output type using `format` argument (`betas` for reliability indices, `rels` for realibility values, `pofs` for probabilify of failure values). Provide confidence levels `cons` and estimator covariance `df_corr` to compute with margin in beta [2].

    Note that the reliability index approach (RIA) is generally less stable than the performance measure approach (PMA). Consider using ``gr.eval_form_pma()`` instead, particularly when using FORM to optimize a design.

    Args:
        model (gr.Model): Model to analyze
        limits (list): Target limit states; must be in model.out; limit state assumed to be critical at g == 0.
        format (str): One of ("betas", "rels", "pofs"). Format for computed reliability information

        cons (dict or None): Target confidence levels;
            key   = limit state name; must be in model.out
            value = confidence level, \in (0, 1)
        df_corr (DataFrame or None): Sampling distribution covariance entries; parameters with no information assumed to be known exactly.
        df_det (DataFrame): Deterministic levels for evaluation; use "nom" for nominal deterministic levels.
        n_maxiter (int): Maximum iterations for each optimization run
        n_restart (int): Number of restarts (== number of optimization runs)
        append (bool): Append MPP results for random values?
        verbose (bool): Print optimization results?

    Returns:
        DataFrame: Results of MPP search

    Notes:
        Since FORM RIA relies on optimization over the limit state, it is often beneficial to scale your limit state to keep values near unity.

    References:
        [1] Tu, Choi, and Park, "A new study on reliability-based design optimization," Journal of Mechanical Design, 1999
        [2] del Rosario, Fenrich, and Iaccarino, "Fast precision margin with the first-order reliability method," AIAA Journal, 2019

    Examples::

        import grama as gr
        from grama.models import make_cantilever_beam
        md_beam = make_cantilever_beam()
        ## Evaluate the reliability of specified designs
        (
            md_beam
            >> gr.ev_form_ria(
                # Specify limit states to analyze
                limits=("g_stress", "g_disp"),
                # Analyze three different thicknesses
                df_det=gr.df_make(t=[2, 3, 4], w=3)
            )
        )

    """
    ## Check invariants
    invariants_eval_model(model)
    invariants_eval_df(df_corr, arg_name="df_corr", acc_none=True)
    invariants_eval_df(df_det, arg_name="df_det", valid_str=["nom"])
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
                df_res = eval_df(model, df=df)
                g = df_res[key].iloc[0]

                return g

            ## Use conservative direction for initial guess
            signs = array(
                [model.density.marginals[k].sign for k in model.var_rand]
            )
            if length(signs) > 0:
                z0 = signs / length(signs)
            else:
                z0 = ones(model.n_var_rand) / sqrt(model.n_var_rand)

            ## Minimize
            res_all = []
            for jnd in range(n_restart):
                res = minimize(
                    fun_jac,
                    z0,
                    args=(),
                    method="SLSQP",
                    jac=True,
                    tol=tol,
                    options={"maxiter": n_maxiter, "disp": False},
                    constraints=[{"type": "eq", "fun": con_limit}],
                )
                # Append only a successful result
                if res["status"] == 0:
                    res_all.append(res)
                # Set a random start; repeat
                z0 = multivariate_normal(
                    [0] * model.n_var_rand, eye(model.n_var_rand)
                )
                z0 = z0 / length(z0)

            # Choose value among restarts
            n_iter_total = sum([res_all[i].nit for i in range(len(res_all))])
            if len(res_all) > 0:
                i_star = argmin([res.fun for res in res_all])
                x_star = res_all[i_star].x
                fun_star = sqrt(res_all[i_star].fun)
                if verbose:
                    print("out = {}: Optimization successful".format(key))
                    print("n_iter = {}".format(res_all[i_star].nit))
                    print("n_iter_total = {}".format(n_iter_total))
            else:
                ## WARNING
                x_star = [NaN] * model.n_var_rand
                fun_star = NaN
                if verbose:
                    print("out = {}: Optimization unsuccessful".format(key))
                    print("n_iter = {}".format(res_all[i_star].nit))
                    print("n_iter_total = {}".format(n_iter_total))

            ## Extract results
            if append:
                df_inner = concat(
                    (
                        df_inner,
                        model.norm2rand(
                            DataFrame(data=[x_star], columns=model.var_rand)
                        ),
                    ),
                    axis=1,
                    sort=False,
                )
            df_inner["beta_" + key] = [fun_star]
            df_return = concat((df_return, df_inner), axis=0, sort=False)

    ## Collapse beta values w/o MPP results
    beta_names = ["beta_" + s for s in limits]
    if not append:
        df_return = (
            df_return.groupby(model.var_det)
            .agg({n: max for n in beta_names})
            .reset_index()
        )

    ## Convert betas if other format requested
    if format == "rels":
        df_return = df_return.apply(
            lambda col: norm.cdf(col) if col.name in beta_names else col
        ).rename(columns={"beta_" + s: "rel_" + s for s in limits})

    elif format == "pofs":
        df_return = df_return.apply(
            lambda col: 1 - norm.cdf(col) if col.name in beta_names else col
        ).rename(columns={"beta_" + s: "pof_" + s for s in limits})

    elif not (format == "betas"):
        print(
            "    format = '{}' unrecognized; returning reliability indices".format(
                format
            )
        )

    return df_return


ev_form_ria = add_pipe(eval_form_ria)
