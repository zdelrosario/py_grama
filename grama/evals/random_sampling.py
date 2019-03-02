import numpy as np
import pandas as pd

from .. import core
from scipy.stats import norm
from toolz import curry

## Simple Monte Carlo
@curry
def eval_monte_carlo(model, n_samples = 1, seed = None, append = True):
    """Evaluates a given model at a given dataframe

    @param n_samples number of Monte Carlo samples to draw
    @param seed random seed to use
    @param append bool flag; append results to original dataframe?

    Only implemented for factorable densities for now.
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

    ## Make space for random draws
    samples = np.zeros((n_samples, model.n_in))

    ## Draw from parameterized distribution family
    for ind in range(model.n_in):
        if model.density.pdf_factors[ind] == "unif":
            samples[:, ind] = \
                np.random.random(n_samples) * (
                    model.density.pdf_param[ind]["upper"] -
                    model.density.pdf_param[ind]["lower"]
                ) + model.density.pdf_param[ind]["lower"]
        elif model.density.pdf_factors[ind] == "norm":
            samples[:, ind] = \
                np.random.normal(
                    size  = n_samples,
                    loc   = model.density.pdf_param[ind]["loc"],
                    scale = model.density.pdf_param[ind]["scale"]
                )

    ## Create dataframe for inputs
    df_inputs = pd.DataFrame(
        data = samples,
        columns = model.domain.inputs
    )

    return core.eval_df(model, df = df_inputs, append = append)
