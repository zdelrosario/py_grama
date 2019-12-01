__all__ = [
    "eval_monte_carlo",
    "ev_monte_carlo",
    "eval_lhs",
    "ev_lhs",
    "eval_sinews",
    "ev_sinews",
    "eval_hybrid",
    "ev_hybrid"
]

import numpy as np
import pandas as pd

from .defaults import eval_df
from ..tools import pipe
from scipy.stats import norm, lognorm
from toolz import curry
from pyDOE import lhs
from numpy.linalg import cholesky, inv

## Simple Monte Carlo
# --------------------------------------------------
@curry
def eval_monte_carlo(model, n=1, df_det=None, seed=None, append=True, skip=False):
    """Evaluates a given model at a given dataframe. Generates outer product
    with deterministic samples

    @param n number of Monte Carlo samples to draw
    @param df_det DataFrame deterministic samples
    @param seed random seed to use
    @param append bool flag; append results to original dataframe?
    @param skip bool flag; skip computing the results? (return the design)
    """
    ## Set seed only if given
    if seed is not None:
        np.random.seed(seed)

    ## Ensure sample count is int
    n = int(n)

    ## Draw samples
    df_quant = pd.DataFrame(
        data=np.random.random((n, model.n_var_rand)),
        columns=model.var_rand
    )

    ## Convert samples to desired marginals
    df_rand = model.var_rand_quantile(df_quant)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        return df_samp
    else:
        return eval_df(model, df=df_samp, append=append)

@pipe
def ev_monte_carlo(*args, **kwargs):
    return eval_monte_carlo(*args, **kwargs)

## Latin Hypercube Sampling (LHS)
# --------------------------------------------------
@curry
def eval_lhs(
        model,
        n=1,
        df_det=None,
        seed=None,
        append=True,
        skip=False,
        criterion=None
):
    """Evaluates a given model on a latin hypercube sample (LHS)
    using the model's density

    @param n number of LHS samples to draw
    @param df_det DataFrame deterministic samples
    @param seed random seed to use
    @param append bool flag; append results to original dataframe?
    @param skip bool flag; skip computing the results? (return the design)
    @param criterion flag for LHS sample criterion
           allowable values: None, "center" ("c"), "maxmin" ("m"),
                             "centermaxmin" ("cm"), "correlation" ("corr")

    Built on pyDOE.lhs

    Only implemented for gaussian copula distributions for now.
    """
    ## Set seed only if given
    if seed is not None:
        np.random.seed(seed)

    ## Ensure sample count is int
    n = int(n)

    ## Draw samples
    df_quant = pd.DataFrame(
        data=lhs(model.n_var_rand, samples=n),
        columns=model.var_rand
    )

    ## Convert samples to desired marginals
    df_rand = model.var_rand_quantile(df_quant)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        return df_samp
    else:
        return eval_df(model, df=df_samp, append=append)

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
        skip=False
):
    """Perform sweeps over each model marginal (sinew)
    """
    ## Set seed only if given
    if seed is not None:
        np.random.seed(seed)

    ## Build quantile sweep data
    q_random = np.tile(
        np.random.random((1, model.n_var_rand, n_sweeps)),
        (n_density, 1, 1)
    )
    q_dense  = np.linspace(0, 1, num=n_density)
    Q_all    = np.zeros(
        (n_density * n_sweeps * model.n_var_rand, model.n_var_rand)
    )
    C_var    = ["tmp"] * (n_density * n_sweeps * model.n_var_rand)
    C_ind    = [0] * (n_density * n_sweeps * model.n_var_rand)

    ## Interlace
    for i_input in range(model.n_var_rand):
        ind_base = i_input * n_density * n_sweeps
        for i_sweep in range(n_sweeps):
            ind_start = ind_base + i_sweep*n_density
            ind_end   = ind_base + (i_sweep+1)*n_density

            Q_all[ind_start:ind_end]          = q_random[:, :, i_sweep]
            Q_all[ind_start:ind_end, i_input] = q_dense
            C_var[ind_start:ind_end] = \
                [model.var_rand[i_input]] * n_density
            C_ind[ind_start:ind_end] = [i_sweep] * n_density

    ## Assemble sampling plan
    df_quant = pd.DataFrame(data=Q_all, columns=model.var_rand)
    df_rand = model.var_rand_quantile(df_quant)
    df_rand[varname] = C_var
    df_rand[indname] = C_ind
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        ## Pass-through
        return df_samp
    else:
        ## Apply
        return eval_df(model, df=df_samp, append=append)

@pipe
def ev_sinews(*args, **kwargs):
    return eval_sinews(*args, **kwargs)

## Hybrid points for Sobol' indices
# --------------------------------------------------
@curry
def eval_hybrid(
        model,
        n_samples=1,
        df_det=None,
        varname="hybrid_var",
        plan="first",
        seed=None,
        append=True,
        skip=False
):
    """Hybrid points for first order indices
    """
    ## Set seed only if given
    if seed is not None:
        np.random.seed(seed)
    n_samples = int(n_samples)

    ## Draw hybrid points
    X = np.random.random((n_samples, model.n_var_rand))
    Z = np.random.random((n_samples, model.n_var_rand))

    ## Reserve space
    Q_all = np.zeros((n_samples * (model.n_var_rand + 1), model.n_var_rand))
    Q_all[:n_samples] = X # Base samples
    C_var = ["_"] * (n_samples * (model.n_var_rand + 1))

    ## Interleave samples
    for i_in in range(model.n_var_rand):
        i_start = (i_in + 1) * n_samples
        i_end   = (i_in + 2) * n_samples

        if plan == "first":
            Q_all[i_start:i_end, :]    = Z
            Q_all[i_start:i_end, i_in] = X[:, i_in]
        elif plan == "total":
            Q_all[i_start:i_end, :]    = X
            Q_all[i_start:i_end, i_in] = Z[:, i_in]
        else:
            raise ValueError("plan must be `first` or `total`")

        C_var[i_start:i_end] = [model.var_rand[i_in]] * n_samples

    ## Construct sampling plan
    df_quant = pd.DataFrame(data=Q_all, columns=model.var_rand)
    ## Convert samples to desired marginals
    df_rand = model.var_rand_quantile(df_quant)
    df_rand[varname] = C_var
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        return df_samp
    else:
        return eval_df(model, df=df_samp, append=append)

@pipe
def ev_hybrid(*args, **kwargs):
    return eval_hybrid(*args, **kwargs)
