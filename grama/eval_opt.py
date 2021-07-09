__all__ = [
    "eval_nls",
    "ev_nls",
    "eval_min",
    "ev_min",
]

from grama import add_pipe, pipe, custom_formatwarning, df_make, \
    eval_df, eval_nominal, eval_monte_carlo, comp_marginals, \
    comp_copula_independence, tran_outer
from numpy import Inf, isfinite
from numpy.random import seed as setseed
from pandas import DataFrame, concat
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.optimize import minimize
from toolz import curry


## Nonlinear least squares
# --------------------------------------------------
@curry
def eval_nls(
    model,
    df_data=None,
    out=None,
    var_fix=None,
    df_init=None,
    append=False,
    tol=1e-6,
    ftol=1e-9,
    gtol=1e-5,
    n_maxiter=100,
    n_restart=1,
    n_process=1,
    method="L-BFGS-B",
    seed=None,
    verbose=True,
):
    r"""Estimate with Nonlinear Least Squares (NLS)

    Estimate best-fit variable levels with nonlinear least squares (NLS).

    Args:
        model (gr.Model): Model to analyze. All model variables
            selected for fitting must be bounded or random. Deterministic
            variables may have semi-infinite bounds.
        df_data (DataFrame): Data for estimating parameters. Variables not
            found in df_data optimized in fitting.
        out (list or None): Output contributions to consider in computing MSE.
            Assumed to be model.out if left as None.
        var_fix (list or None): Variables to fix to nominal levels. Note that
            variables with domain width zero will automatically be fixed.
        df_init (DataFrame): Initial guesses for parameters; overrides n_restart
        append (bool): Append metadata? (Initial guess, MSE, optimizer status)
        tol (float): Optimizer convergence tolerance
        n_maxiter (int): Optimizer maximum iterations
        n_restart (int): Number of restarts; beyond n_restart=1 random
            restarts are used.
        seed (int OR None): Random seed for restarts
        verbose (bool): Print messages to console?

    Returns:
        DataFrame: Results of estimation

    Examples:
        >>> import grama as gr
        >>> from grama.data import df_trajectory_full
        >>> from grama.models import make_trajectory_linear
        >>>
        >>> md_trajectory = make_trajectory_linear()
        >>>
        >>> df_fit = (
        >>>     md_trajectory
        >>>     >> gr.ev_nls(df_data=df_trajectory_full)
        >>> )
        >>>
        >>> print(df_fit)

    """
    ## Check `out` invariants
    if out is None:
        out = model.out
        if verbose:
            print("... eval_nls setting out = {}".format(out))
    set_diff = set(out).difference(set(df_data.columns))
    if len(set_diff) > 0:
        raise ValueError(
            "out must be subset of df_data.columns\n"
            + "difference = {}".format(set_diff)
        )

    ## Determine variables to be fixed
    if var_fix is None:
        var_fix = set()
    else:
        var_fix = set(var_fix)
    for var in model.var_det:
        wid = model.domain.get_width(var)
        if wid == 0:
            var_fix.add(var)
    if verbose:
        print("... eval_nls setting var_fix = {}".format(list(var_fix)))

    ## Determine variables for evaluation
    var_feat = set(model.var).intersection(set(df_data.columns))
    if verbose:
        print("... eval_nls setting var_feat = {}".format(list(var_feat)))

    ## Determine variables for fitting
    var_fit = set(model.var).difference(var_fix.union(var_feat))
    if len(var_fit) == 0:
        raise ValueError(
            "No var selected for fitting!\n"
            + "Try checking model bounds and df_data.columns."
        )

    ## Separate var_fit into det and rand
    var_fit_det = list(set(model.var_det).intersection(var_fit))
    var_fit_rand = list(set(model.var_rand).intersection(var_fit))

    ## Construct bounds, fix var_fit order
    var_fit = var_fit_det + var_fit_rand
    bounds = []
    var_prob = []
    for var in var_fit_det:
        if not isfinite(model.domain.get_nominal(var)):
            var_prob.append(var)
        bounds.append(model.domain.get_bound(var))
    if len(var_prob) > 0:
        raise ValueError(
            "all variables to be fitted must finite nominal value\n"
            + "offending var = {}".format(var_prob)
        )

    for var in var_fit_rand:
        bounds.append(
            (model.density.marginals[var].q(0), model.density.marginals[var].q(1),)
        )

    ## Determine initial guess points
    df_nom = eval_nominal(model, df_det="nom", skip=True)

    ## Use specified initial guess(es)
    if not (df_init is None):
        # Check invariants
        set_diff = set(var_fit).difference(set(df_init.columns))
        if len(set_diff) > 0:
            raise ValueError(
                "var_fit must be subset of df_init.columns\n"
                + "difference = {}".format(set_diff)
            )
        # Pull n_restart
        n_restart = df_init.shape[0]

    ## Generate initial guess(es)
    else:

        df_init = df_nom[var_fit]
        if n_restart > 1:
            if not (seed is None):
                setseed(seed)
            ## Collect sweep-able deterministic variables
            var_sweep = list(
                filter(
                    lambda v: isfinite(model.domain.get_width(v))
                    & (model.domain.get_width(v) > 0),
                    model.var_det,
                )
            )
            ## Generate pseudo-marginals
            dicts_var = {}
            for v in var_sweep:
                dicts_var[v] = {
                    "dist": "uniform",
                    "loc": model.domain.get_bound(v)[0],
                    "scale": model.domain.get_width(v),
                }
            ## Overwrite model
            md_sweep = comp_marginals(model, **dicts_var)
            md_sweep = comp_copula_independence(md_sweep)
            ## Generate random start points
            df_rand = eval_monte_carlo(
                md_sweep, n=n_restart - 1, df_det="nom", skip=True,
            )
            df_init = concat((df_init, df_rand[var_fit]), axis=0).reset_index(drop=True)

    ## Iterate over initial guesses
    df_res = DataFrame()
    def fun_mp(i):
        x0 = df_init[var_fit].iloc[i].values
        ## Build evaluator
        def objective(x):
            """x = [var_fit]"""
            ## Evaluate model
            df_var = tran_outer(
                df_data[var_feat],
                concat(
                    (df_nom[var_fix].iloc[[0]], df_make(**dict(zip(var_fit, x)))),
                    axis=1,
                ),
            )
            df_tmp = eval_df(model, df=df_var)

            ## Compute joint MSE
            return ((df_tmp[out].values - df_data[out].values) ** 2).mean()

        ## Run optimization
        res = minimize(
            objective,
            x0,
            args=(),
            method=method,
            jac=False,
            tol=tol,
            options={"maxiter": n_maxiter, "disp": False, "ftol": ftol, "gtol": gtol,},
            bounds=bounds,
        )

        df_tmp = df_make(
            **dict(zip(var_fit, res.x)),
            **dict(zip(map(lambda s: s + "_0", var_fit), x0)),
        )
        df_tmp["success"] = [res.success]
        df_tmp["message"] = [res.message]
        df_tmp["n_iter"] = [res.nit]
        df_tmp["mse"] = [res.fun]
        return df_tmp

    num_process = n_process
    if num_process > n_restart:
        num_process = n_restart

    pool = Pool(processes = num_process)
    mp_out = pool.map(fun_mp, range(n_restart))
    df_res = concat(mp_out, axis=0,).reset_index(drop=True)

    ## Post-process
    if append:
        return df_res
    return df_res[var_fit]


ev_nls = add_pipe(eval_nls)

## Minimize
# --------------------------------------------------
@curry
def eval_min(
    model,
    out_min=None,
    out_geq=None,
    out_leq=None,
    out_eq=None,
    method="SLSQP",
    tol=1e-6,
    n_restart=1,
    n_maxiter=50,
    seed=None,
    df_start=None,
):
    r"""Constrained minimization using functions from a model

    Perform constrained minimization using functions from a model. Model must
    have deterministic variables only.

    Wrapper for scipy.optimize.minimize

    Args:
        model (gr.Model): Model to analyze. All model variables must be
            deterministic.
        out_min (str): Output to use as minimization objective.
        out_geq (None OR list of str): Outputs to use as geq constraints; out >= 0
        out_leq (None OR list of str): Outputs to use as leq constraints; out <= 0
        out_eq (None OR list of str): Outputs to use as equality constraints; out == 0

        method (str): Optimization method; see the documentation for
            scipy.optimize.minimize for options.
        tol (float): Optimization objective convergence tolerance
        n_restart (int): Number of restarts; beyond n_restart=1 random
            restarts are used.
        df_start (None or DataFrame): Specific starting values to use; overrides
            n_restart if non None provided.

    Returns:
        DataFrame: Results of optimization

    Examples:
        >>> import grama as gr
        >>> md = (
        >>>     gr.Model("Constrained Rosenbrock")
        >>>     >> gr.cp_function(
        >>>         fun=lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2,
        >>>         var=["x", "y"],
        >>>         out=["c"],
        >>>     )
        >>>     >> gr.cp_function(
        >>>         fun=lambda x: (x[0] - 1)**3 - x[1] + 1,
        >>>         var=["x", "y"],
        >>>         out=["g1"],
        >>>     )
        >>>     >> gr.cp_function(
        >>>         fun=lambda x: x[0] + x[1] - 2,
        >>>         var=["x", "y"],
        >>>         out=["g2"],
        >>>     )
        >>>     >> gr.cp_bounds(
        >>>         x=(-1.5, +1.5),
        >>>         y=(-0.5, +2.5),
        >>>     )
        >>> )
        >>> md >> gr.ev_min(
        >>>     out_min="c",
        >>>     out_leq=["g1", "g2"]
        >>> )

    """
    ## Check that model has only deterministic variables
    if model.n_var_rand > 0:
        raise ValueError("model must have no random variables")
    ## Check that objective is in model
    if not (out_min in model.out):
        raise ValueError("model must contain out_min")
    ## Check that constraints are in model
    if not (out_geq is None):
        out_diff = set(out_geq).difference(set(model.out))
        if len(out_diff) > 0:
            raise ValueError(
                "model must contain each out_geq; missing {}".format(out_diff)
            )
    if not (out_leq is None):
        out_diff = set(out_leq).difference(set(model.out))
        if len(out_diff) > 0:
            raise ValueError(
                "model must contain each out_leq; missing {}".format(out_diff)
            )
    if not (out_eq is None):
        out_diff = set(out_eq).difference(set(model.out))
        if len(out_diff) > 0:
            raise ValueError(
                "model must contain each out_eq; missing {}".format(out_diff)
            )

    ## Formulate initial guess
    df_nom = eval_nominal(model, df_det="nom", skip=True)
    if df_start is None:
        df_start = df_nom[model.var]

        if n_restart > 1:
            if not (seed is None):
                setseed(seed)
            ## Collect sweep-able deterministic variables
            var_sweep = list(
                filter(
                    lambda v: isfinite(model.domain.get_width(v))
                    & (model.domain.get_width(v) > 0),
                    model.var_det,
                )
            )
            ## Generate pseudo-marginals
            dicts_var = {}
            for v in var_sweep:
                dicts_var[v] = {
                    "dist": "uniform",
                    "loc": model.domain.get_bound(v)[0],
                    "scale": model.domain.get_width(v),
                }
            ## Overwrite model
            md_sweep = comp_marginals(model, **dicts_var)
            md_sweep = comp_copula_independence(md_sweep)
            ## Generate random start points
            df_rand = eval_monte_carlo(
                md_sweep, n=n_restart - 1, df_det="nom", skip=True,
            )
            df_start = concat((df_start, df_rand[model.var]), axis=0).reset_index(
                drop=True
            )
    else:
        n_restart = df_start.shape[0]

    ## Factory for wrapping model's output
    def make_fun(out, sign=+1):
        def fun(x):
            df = DataFrame([x], columns=model.var)
            df_res = eval_df(model, df)
            return sign * df_res[out]

        return fun

    ## Create helper functions for constraints
    constraints = []

    if not (out_geq is None):
        for out in out_geq:
            constraints.append(
                {"type": "ineq", "fun": make_fun(out),}
            )

    if not (out_leq is None):
        for out in out_leq:
            constraints.append(
                {"type": "ineq", "fun": make_fun(out, sign=-1),}
            )

    if not (out_eq is None):
        for out in out_eq:
            constraints.append(
                {"type": "eq", "fun": make_fun(out),}
            )

    ## Parse the bounds for minimize
    bounds = list(map(lambda k: model.domain.bounds[k], model.var))

    ## Run optimization
    df_res = DataFrame()
    for i in range(n_restart):
        x0 = df_start[model.var].iloc[i].values
        res = minimize(
            make_fun(out_min),
            x0,
            args=(),
            method=method,
            jac=False,
            tol=tol,
            options={"maxiter": n_maxiter, "disp": False},
            constraints=constraints,
            bounds=bounds,
        )

        df_opt = df_make(
            **dict(zip(model.var, res.x)),
            **dict(zip(map(lambda s: s + "_0", model.var), x0)),
        )
        df_tmp = eval_df(model, df=df_opt)
        df_tmp["success"] = [res.success]
        df_tmp["message"] = [res.message]
        df_tmp["n_iter"] = [res.nit]

        df_res = concat((df_res, df_tmp), axis=0).reset_index(drop=True)

    return df_res


ev_min = add_pipe(eval_min)
