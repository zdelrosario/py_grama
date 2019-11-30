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

from .. import core
from ..core import pipe
from scipy.stats import norm, lognorm
from toolz import curry
from pyDOE import lhs
from numpy.linalg import cholesky, inv

## Simple Monte Carlo
# --------------------------------------------------
@curry
def eval_monte_carlo(model, n_samples=1, seed=None, append=True):
    """Evaluates a given model at a given dataframe

    @param n_samples number of Monte Carlo samples to draw
    @param seed random seed to use
    @param append bool flag; append results to original dataframe?

    Only implemented for gaussian copula distributions for now.
    """

    ## Check if distribution is valid
    if model.density is not None:
        if len(model.density.pdf_factors) != len(model.density.pdf_param):
            raise ValueError("model.density.pdf_factors not same length as model.density.pdf_param!")
    else:
        raise ValueError("Model density must be factorable!")

    ## Set seed only if given
    if seed is not None:
        np.random.seed(seed)

    ## Draw samples
    quantiles = np.random.random((n_samples, model.n_in))

    ## Convert samples to desired marginals
    samples = model.sample_quantile(quantiles)

    ## Create dataframe for inputs
    df_inputs = pd.DataFrame(
        data = samples,
        columns = model.domain.inputs
    )

    return core.eval_df(model, df = df_inputs, append = append)

@pipe
def ev_monte_carlo(*args, **kwargs):
    return eval_monte_carlo(*args, **kwargs)

## Latin Hypercube Sampling (LHS)
# --------------------------------------------------
@curry
def eval_lhs(model, n_samples=1, seed=None, append=True, criterion=None):
    """Evaluates a given model on a latin hypercube sample (LHS)
    using the model's density

    @param n_samples number of LHS samples to draw
    @param seed random seed to use
    @param append bool flag; append results to original dataframe?
    @param criterion flag for LHS sample criterion
           allowable values: None, "center" ("c"), "maxmin" ("m"),
                             "centermaxmin" ("cm"), "correlation" ("corr")

    Built on pyDOE.lhs

    Only implemented for gaussian copula distributions for now.
    """

    ## Check if distribution is valid
    if model.density is not None:
        if len(model.density.pdf_factors) != len(model.density.pdf_param):
            raise ValueError("model.density.pdf_factors not same length as model.density.pdf_param!")
    else:
        raise ValueError("Model density must be factorable!")

    ## Set seed only if given
    if seed is not None:
        np.random.seed(seed)

    ## Draw samples
    quantiles = lhs(model.n_in, samples=n_samples)

    ## Convert samples to desired marginals
    samples = model.sample_quantile(quantiles)

    ## Create dataframe for inputs
    df_inputs = pd.DataFrame(
        data = samples,
        columns = model.domain.inputs
    )

    return core.eval_df(model, df = df_inputs, append = append)

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
        varname="sweep_var",
        indname="sweep_ind",
        append=True
):
    """Perform sweeps over each model marginal (sinew)
    """
    ## Set seed only if given
    if seed is not None:
        np.random.seed(seed)

    ## Build quantile sweep data
    q_random = np.tile(
        np.random.random((1, model.n_in, n_sweeps)),
        (n_density, 1, 1)
    )
    q_dense  = np.linspace(0, 1, num=n_density)
    Q_all    = np.zeros((n_density * n_sweeps * model.n_in, model.n_in))
    C_var    = ["tmp"] * (n_density * n_sweeps * model.n_in)
    C_ind    = [0] * (n_density * n_sweeps * model.n_in)

    ## Interlace
    for i_input in range(model.n_in):
        ind_base = i_input * n_density * n_sweeps
        for i_sweep in range(n_sweeps):
            ind_start = ind_base + i_sweep*n_density
            ind_end   = ind_base + (i_sweep+1)*n_density

            Q_all[ind_start:ind_end]          = q_random[:, :, i_sweep]
            Q_all[ind_start:ind_end, i_input] = q_dense
            C_var[ind_start:ind_end] = \
                [model.domain.inputs[i_input]] * n_density
            C_ind[ind_start:ind_end] = [i_sweep] * n_density

    ## Apply
    samples = model.sample_quantile(Q_all)
    df_inputs = pd.DataFrame(data=samples, columns=model.domain.inputs)
    df_result = core.eval_df(model, df=df_inputs, append=append)

    df_result[varname] = C_var
    df_result[indname] = C_ind

    return df_result

@pipe
def ev_sinews(*args, **kwargs):
    return eval_sinews(*args, **kwargs)

## Hybrid points for Sobol' indices
# --------------------------------------------------
@curry
def eval_hybrid(
        model,
        n_samples=1,
        varname="hybrid_var",
        plan="first",
        seed=None,
        append=True
):
    """Hybrid points for first order indices
    """
    ## Set seed only if given
    if seed is not None:
        np.random.seed(seed)
    n_samples = int(n_samples)

    ## Draw hybrid points
    X = np.random.random((n_samples, model.n_in))
    Z = np.random.random((n_samples, model.n_in))

    ## Reserve space
    Q_all = np.zeros((n_samples * (model.n_in + 1), model.n_in))
    Q_all[:n_samples] = X # Base samples
    C_var = ["_"] * (n_samples * (model.n_in + 1))

    ## Interleave samples
    for i_in in range(model.n_in):
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

        C_var[i_start:i_end] = [model.domain.inputs[i_in]] * n_samples

    ## Apply
    samples = model.sample_quantile(Q_all)
    df_inputs = pd.DataFrame(data=samples, columns=model.domain.inputs)
    df_result = core.eval_df(model, df=df_inputs, append=append)

    df_result[varname] = C_var

    return df_result

@pipe
def ev_hybrid(*args, **kwargs):
    return eval_hybrid(*args, **kwargs)
