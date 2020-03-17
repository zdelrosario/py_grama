__all__ = [
    "eval_monte_carlo",
    "ev_monte_carlo",
    "eval_lhs",
    "ev_lhs",
    "eval_sinews",
    "ev_sinews",
    "eval_hybrid",
    "ev_hybrid",
]

from numpy import tile, linspace, zeros, isfinite
from numpy.random import random
from numpy.random import seed as set_seed
from pandas import DataFrame

import warnings

import grama as gr
from grama import pipe, custom_formatwarning
from scipy.stats import norm, lognorm
from toolz import curry
from pyDOE import lhs
from numpy.linalg import cholesky, inv
from numbers import Integral

warnings.formatwarning = custom_formatwarning

## Simple Monte Carlo
# --------------------------------------------------
@curry
def eval_monte_carlo(model, n=1, df_det=None, seed=None, append=True, skip=False):
    r"""Monte Carlo evaluation

    Evaluates a given model at a given dataframe. Generates outer product
    with deterministic samples.

    Args:
        model (gr.Model): Model to evaluate
        n (numeric): number of Monte Carlo samples to draw
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        seed (int): random seed to use
        append (bool): Append results to random values?
        skip (bool): Skip evaluation of the functions?

    Returns:
        DataFrame: Results of evaluation or unevaluated design

    Examples:

        >>> import grama as gr
        >>> from grama.models import make_test
        >>> md = make_test()
        >>> df = md >> gr.ev_monte_carlo(n=1e2, df_det="nom")
        >>> df.describe()

    """
    ## Set seed only if given
    if seed is not None:
        set_seed(seed)

    ## Ensure sample count is int
    if not isinstance(n, Integral):
        print("eval_monte_carlo() is rounding n...")
        n = int(n)

    ## Draw samples
    df_rand = model.density.sample(n=n, seed=seed)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        ## Evaluation estimate
        runtime_est = model.runtime(df_samp.shape[0])
        if runtime_est > 0:
            print(
                "Estimated runtime for design with model ({0:1}):\n  {1:4.3} sec".format(
                    model.name, runtime_est
                )
            )
        else:
            print("Design runtime estimates unavailable; model has no timing data.")

        ## Attach metadata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_samp._plot_info = {
                "type": "monte_carlo_inputs",
                "var": model.var_rand,
            }

        return df_samp
    else:
        df_res = gr.eval_df(model, df=df_samp, append=append)

        ## Attach metadata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_res._plot_info = {"type": "monte_carlo_outputs", "out": model.out}

        return df_res


@pipe
def ev_monte_carlo(*args, **kwargs):
    return eval_monte_carlo(*args, **kwargs)


## Latin Hypercube Sampling (LHS)
# --------------------------------------------------
@curry
def eval_lhs(
    model, n=1, df_det=None, seed=None, append=True, skip=False, criterion=None
):
    r"""Latin Hypercube evaluation

    Evaluates a given model on a latin hypercube sample (LHS) using the model's
    density.

    Args:
        model (gr.Model): Model to evaluate
        n (numeric): Number of LHS samples to draw
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        seed (int): Random seed to use
        append (bool): Append results to conservative inputs?
        skip (bool): Skip evaluation of the functions?
        criterion (str): flag for LHS sample criterion
            allowable values: None, "center" ("c"), "maxmin" ("m"),
            "centermaxmin" ("cm"), "correlation" ("corr")

    Returns:
        DataFrame: Results of evaluation or unevaluated design

    Notes:
        - Wrapper on pyDOE.lhs

    """
    ## Set seed only if given
    if seed is not None:
        set_seed(seed)

    ## Ensure sample count is int
    if not isinstance(n, Integral):
        print("eval_lhs() is rounding n...")
        n = int(n)

    ## Draw samples
    df_quant = DataFrame(data=lhs(model.n_var_rand, samples=n), columns=model.var_rand)

    ## Convert samples to desired marginals
    df_rand = model.density.pr2sample(df_quant)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        return df_samp
    else:
        return gr.eval_df(model, df=df_samp, append=append)


@pipe
def ev_lhs(*args, **kwargs):
    return eval_lhs(*args, **kwargs)


## Marginal sweeps with random origins
# --------------------------------------------------
@curry
def eval_sinews(
    model,
    n_density=10,
    n_sweeps=3,
    seed=None,
    df_det=None,
    varname="sweep_var",
    indname="sweep_ind",
    append=True,
    skip=False,
):
    r"""Sweep study

    Perform coordinate sweeps over each model random variable ("sinew" design).
    Use random starting points drawn from the joint density.

    Use gr.plot_auto() to construct a quick visualization of the output
    dataframe. Use `skip` version to visualize the design, and non-skipped
    version to visualize the results.

    Args:
        model (gr.Model): Model to evaluate
        n_density (numeric): Number of points along each sweep
        n_sweeps (numeric): Number of sweeps per-random variable
        seed (int): Random seed to use
        df_det (DataFrame): Deterministic levels for evaluation;
            use "nom" for nominal deterministic levels,
            use "swp" to sweep deterministic variables
        varname (str): Column name to give for sweep variable; default="sweep_var"
        indname (str): Column name to give for sweep index; default="sweep_ind"
        append (bool): Append results to conservative inputs?
        skip (bool): Skip evaluation of the functions?

    Returns:
        DataFrame: Results of evaluation or unevaluated design

    Examples:

        >>> import grama as gr
        >>> md = gr.make_cantilever_beam()
        >>> # Skip evaluation, vis. design
        >>> df_design = md >> gr.ev_sinews(df_det="nom", skip=True)
        >>> df_design >> gr.pt_auto()
        >>> # Vis results
        >>> df_sinew = md >> gr.ev_sinews(df_det="nom")
        >>> df_sinew >> gr.pt_auto()

    """
    ## Override model if deterministic sweeps desired
    if df_det == "swp":
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
        model = gr.comp_marginals(model, **dicts_var)
        ## Restore flag
        df_det = "nom"

    ## Set seed only if given
    if seed is not None:
        set_seed(seed)

    ## Ensure sample count is int
    if not isinstance(n_density, Integral):
        print("eval_sinews() is rounding n_density...")
        n_density = int(n_density)
    if not isinstance(n_sweeps, Integral):
        print("eval_sinews() is rounding n_sweeps...")
        n_sweeps = int(n_sweeps)

    ## Build quantile sweep data
    q_random = tile(random((1, model.n_var_rand, n_sweeps)), (n_density, 1, 1))
    q_dense = linspace(0, 1, num=n_density)
    Q_all = zeros((n_density * n_sweeps * model.n_var_rand, model.n_var_rand))
    C_var = ["tmp"] * (n_density * n_sweeps * model.n_var_rand)
    C_ind = [0] * (n_density * n_sweeps * model.n_var_rand)

    ## Interlace
    for i_input in range(model.n_var_rand):
        ind_base = i_input * n_density * n_sweeps
        for i_sweep in range(n_sweeps):
            ind_start = ind_base + i_sweep * n_density
            ind_end = ind_base + (i_sweep + 1) * n_density

            Q_all[ind_start:ind_end] = q_random[:, :, i_sweep]
            Q_all[ind_start:ind_end, i_input] = q_dense
            C_var[ind_start:ind_end] = [model.var_rand[i_input]] * n_density
            C_ind[ind_start:ind_end] = [i_sweep] * n_density

            ## Modify endpoints for infinite support
            if not isfinite(model.density.marginals[model.var_rand[i_input]].q(0)):
                Q_all[ind_start, i_input] = 1 / n_density / 10
            if not isfinite(model.density.marginals[model.var_rand[i_input]].q(1)):
                Q_all[ind_end - 1, i_input] = 1 - 1 / n_density / 10

    ## Assemble sampling plan
    df_pr = DataFrame(data=Q_all, columns=model.var_rand)
    df_rand = model.density.pr2sample(df_pr)
    df_rand[varname] = C_var
    df_rand[indname] = C_ind
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        ## Evaluation estimate
        runtime_est = model.runtime(df_samp.shape[0])
        if runtime_est > 0:
            print(
                "Estimated runtime for design with model ({0:1}):\n  {1:4.3} sec".format(
                    model.name, runtime_est
                )
            )
        else:
            print("Design runtime estimates unavailable; model has no timing data.")

        ## For autoplot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_samp._plot_info = {"type": "sinew_inputs", "var": model.var_rand}

        ## Pass-through
        return df_samp
    else:
        ## Apply
        df_res = gr.eval_df(model, df=df_samp, append=append)
        ## For autoplot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_res._plot_info = {
                "type": "sinew_outputs",
                "var": model.var_rand,
                "out": model.out,
            }

        return df_res


@pipe
def ev_sinews(*args, **kwargs):
    return eval_sinews(*args, **kwargs)


## Hybrid points for Sobol' indices
# --------------------------------------------------
@curry
def eval_hybrid(
    model,
    n=1,
    plan="first",
    df_det=None,
    varname="hybrid_var",
    seed=None,
    append=True,
    skip=False,
):
    r"""Hybrid points for Sobol' indices

    Use the "hybrid point" design (Sobol', 1999) to support estimating Sobol'
    indices. Use gr.tran_sobol() to post-process the results and compute
    estimates.

    Args:
        model (gr.Model): Model to evaluate; must have CopulaIndependence
        n (numeric): Number of points along each sweep
        plan (str): Sobol' index to compute; plan={"first", "total"}
        seed (int): Random seed to use
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        varname (str): Column name to give for sweep variable; default="hybrid_var"
        append (bool): Append results to conservative inputs?
        skip (bool): Skip evaluation of the functions?

    Returns:
        DataFrame: Results of evaluation or unevaluated design

    References:
        I.M. Sobol', "Sensitivity Estimates for Nonlinear Mathematical Models"
        (1999) MMCE, Vol 1.

    Examples:

        >>> import grama as gr
        >>> md = gr.make_cantilever_beam()
        >>> df_first = md >> gr.ev_hybrid(df_det="nom", plan="first")
        >>> df_first >> gr.tf_sobol()
        >>>
        >>> df_total = md >> gr.ev_hybrid(df_det="nom", plan="total")
        >>> df_total >> gr.tf_sobol()

    """
    ## Check invariants
    if not isinstance(model.density.copula, gr.CopulaIndependence):
        raise ValueError(
            "model must have CopulaIndependence structure;\n"
            + "Sobol' indices only defined for independent variables"
        )

    ## Set seed only if given
    if seed is not None:
        set_seed(seed)

    if not isinstance(n, Integral):
        print("eval_hybrid() is rounding n...")
        n = int(n)

    ## Draw hybrid points
    X = random((n, model.n_var_rand))
    Z = random((n, model.n_var_rand))

    ## Reserve space
    Q_all = zeros((n * (model.n_var_rand + 1), model.n_var_rand))
    Q_all[:n] = X  # Base samples
    C_var = ["_"] * (n * (model.n_var_rand + 1))

    ## Interleave samples
    for i_in in range(model.n_var_rand):
        i_start = (i_in + 1) * n
        i_end = (i_in + 2) * n

        if plan == "first":
            Q_all[i_start:i_end, :] = Z
            Q_all[i_start:i_end, i_in] = X[:, i_in]
        elif plan == "total":
            Q_all[i_start:i_end, :] = X
            Q_all[i_start:i_end, i_in] = Z[:, i_in]
        else:
            raise ValueError("plan must be `first` or `total`")

        C_var[i_start:i_end] = [model.var_rand[i_in]] * n

    ## Construct sampling plan
    df_pr = DataFrame(data=Q_all, columns=model.var_rand)
    ## Convert samples to desired marginals
    df_rand = model.density.pr2sample(df_pr)
    df_rand[varname] = C_var
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_samp._meta = dict(
                type="eval_hybrid",
                varname=varname,
                plan=plan,
                var_rand=model.var_rand,
                out=model.out,
            )

        return df_samp
    else:
        df_res = gr.eval_df(model, df=df_samp, append=append)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_res._meta = dict(
                type="eval_hybrid",
                varname=varname,
                plan=plan,
                var_rand=model.var_rand,
                out=model.out,
            )

        return df_res


@pipe
def ev_hybrid(*args, **kwargs):
    return eval_hybrid(*args, **kwargs)
