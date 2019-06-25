import numpy as np
import pandas as pd

from .. import core
from toolz import curry

## Nominal evaluation
@curry
def ev_nominal(model, append = True):
    """Evaluates a given model at a model nominal conditions

    @param n_samples number of Monte Carlo samples to draw
    @param seed random seed to use
    @param append bool flag; append results to nominal inputs?

    Only implemented for gaussian copula distributions for now.
    """

    ## Check if distribution is valid
    if model.density is not None:
        if len(model.density.pdf_factors) != len(model.density.pdf_param):
            raise ValueError("model.density.pdf_factors not same length as model.density.pdf_param!")
    else:
        raise ValueError("Model density must be factorable!")

    ## Draw from underlying gaussian
    quantiles = np.ones((1, model.n_in)) * 0.5 # Median

    ## Convert samples to desired marginals
    samples = model.sample_quantile(quantiles)

    ## Create dataframe for inputs
    df_inputs = pd.DataFrame(
        data = samples,
        columns = model.domain.inputs
    )

    return core.ev_df(model, df = df_inputs, append = append)
