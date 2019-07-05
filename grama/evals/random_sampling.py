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
    quantiles = lhs(model.n_in, samples = n_samples)

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
