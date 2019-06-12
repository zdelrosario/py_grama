import numpy as np
import pandas as pd

from .. import core
from scipy.stats import norm, lognorm
from toolz import curry

## Simple Monte Carlo
@curry
def eval_monte_carlo(model, n_samples = 1, seed = None, append = True):
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

    ## Draw from underlying gaussian
    if model.density.pdf_corr is not None:
        ## Build correlation structure
        Sigma = np.eye(model.n_in)
        Sigma[np.triu_indices(model.n_in, 1)] = model.density.pdf_corr
        Sigma = Sigma + (Sigma - np.eye(model.n_in)).T
        ## Draw samples
        gaussian_samples = np.random.multivariate_normal(
            mean = np.zeros(model.n_in),
            cov  = Sigma,
            size = n_samples
        )
        ## Convert to uniform marginals
        quantiles = norm.cdf(gaussian_samples)
    ## Skip if no dependence structure
    else:
        quantiles = np.random.random((n_samples, model.n_in))

    ## Convert samples to desired marginals
    samples = model.sample_quantile(quantiles)

    ## Create dataframe for inputs
    df_inputs = pd.DataFrame(
        data = samples,
        columns = model.domain.inputs
    )

    return core.eval_df(model, df = df_inputs, append = append)
