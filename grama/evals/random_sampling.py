import numpy as np
import pandas as pd

from .. import core
from scipy.stats import norm, lognorm
from toolz import curry
from pyDOE import lhs
from numpy.linalg import cholesky, inv

## Simple Monte Carlo
@curry
def ev_monte_carlo(model, n_samples = 1, seed = None, append = True):
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

    return core.ev_df(model, df = df_inputs, append = append)

## Latin Hypercube Sampling (LHS)
@curry
def ev_lhs(model, n_samples = 1, seed = None, append = True, criterion = None):
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
    quantiles = lhs(model.n_in, samples = n_samples)

    ## Convert samples to desired marginals
    samples = model.sample_quantile(quantiles)

    ## Create dataframe for inputs
    df_inputs = pd.DataFrame(
        data = samples,
        columns = model.domain.inputs
    )

    return core.ev_df(model, df = df_inputs, append = append)

## Marginal sweeps with random origins
@curry
def ev_sweeps_marginal(model, n_density=10, n_sweeps=3, append=True):
    """Perform sweeps over each model marginal
    """
    ## Build quantile sweep data
    q_random = np.random.random((n_density, model.n_in, n_sweeps))
    q_dense  = np.linspace(0, 1, num=n_density)
    Q_all    = np.zeros((n_density * n_sweeps * model.n_in, model.n_in))
    C_labels = ["tmp"] * (n_density * n_sweeps * model.n_in)

    ## Interlace
    for i_input in range(model.n_in):
        ind_base = i_input * n_density * n_sweeps
        for i_sweep in range(n_sweeps):
            ind_start = ind_base + i_sweep*n_density
            ind_end   = ind_base + (i_sweep+1)*n_density

            Q_all[ind_start:ind_end]          = q_random[:, :, i_sweep]
            Q_all[ind_start:ind_end, i_input] = q_dense
            C_labels[ind_start:ind_end] = \
                [model.domain.inputs[i_input] + "_s" + str(i_sweep)] * \
                n_density

    ## Apply
    df_inputs = pd.DataFrame(data=Q_all, columns=model.domain.inputs)
    df_result = core.ev_df(model, df=df_inputs, append=append)
    df_result["sweep_var"] = C_labels

    return df_result
