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
        Sigma[np.tril_indices(model.n_in,-1)] = model.density.pdf_corr
        ## Draw samples
        gaussian_samples = np.random.multivariate_normal(
            mean = np.zeros(model.n_in),
            cov  = Sigma,
            size = n_samples
        )
        ## Convert to uniform marginals
        samples = norm.cdf(gaussian_samples)
    ## Skip if no dependence structure
    else:
        samples = np.random.random((n_samples, model.n_in))

    ## Convert samples to desired marginals
    ## TODO: More programmatic way of accessing marginal transforms....
    for ind in range(model.n_in):
        if model.density.pdf_factors[ind] == "unif":
            samples[:, ind] = \
                samples[:, ind] * (
                    model.density.pdf_param[ind]["upper"] -
                    model.density.pdf_param[ind]["lower"]
                ) + model.density.pdf_param[ind]["lower"]
        elif model.density.pdf_factors[ind] == "norm":
            samples[:, ind] = \
                norm.ppf(
                    samples[:, ind],
                    loc   = model.density.pdf_param[ind]["loc"],
                    scale = model.density.pdf_param[ind]["scale"]
                )
        elif model.density.pdf_factors[ind] == "lognorm":
            samples[:, ind] = \
                lognorm.ppf(
                    samples[:, ind],
                    s     = model.density.pdf_param[ind]["s"],
                    loc   = model.density.pdf_param[ind]["loc"],
                    scale = model.density.pdf_param[ind]["scale"]
                )
        else:
            raise ValueError("model.density.pdf_factors[{}] nor supported".format(ind))

    ## Create dataframe for inputs
    df_inputs = pd.DataFrame(
        data = samples,
        columns = model.domain.inputs
    )

    return core.eval_df(model, df = df_inputs, append = append)
