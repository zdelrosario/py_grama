import numpy as np
import pandas as pd

from .. import core
from toolz import curry

## Nominal evaluation
@curry
def ev_nominal(model, append = True):
    """Evaluates a given model at a model nominal conditions (median)

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

## Conservative quantile evaluation
@curry
def ev_conservative(model, quantiles = None, append = True):
    """Evaluates a given model at conservative input quantiles

    Uses model specifications to determine the "conservative" direction
    for each input, and evaluates the model at the desired quantile.
    Provided primarily for comparing UQ against pseudo-deterministic
    design criteria.

    Note that if there is no conservative direction for the given input,
    the given quantile will be ignored and the median will automatically
    be selected.

    @param quantiles array of integer values in [0, 0.5]; assumed to be
    a lower quantile, automatically corrected if the upper quantile is
    conservative.
    @param append bool flag; append results to nominal inputs?

    @pre (len(quantiles) == model.n_in) || (quantiles is None)

    Only implemented for gaussian copula distributions for now.
    """
    if quantiles is None:
        quantiles = [0.01] * model.n_in

    ## Check if distribution is valid
    if model.density is not None:
        if len(model.density.pdf_factors) != len(model.density.pdf_param):
            raise ValueError("model.density.pdf_factors not same length as model.density.pdf_param!")
    else:
        raise ValueError("Model density must be factorable!")

    ## Modify quantiles for conservative directions
    quantiles = [
        0.5 + (0.5 - quantiles[i]) * model.density.pdf_qt_sign[i] \
        for i in range(model.n_in)
    ]
    quantiles = np.atleast_2d(quantiles)

    ## Convert samples to desired marginals
    samples = model.sample_quantile(quantiles)

    ## Create dataframe for inputs
    df_inputs = pd.DataFrame(
        data = samples,
        columns = model.domain.inputs
    )

    return core.ev_df(model, df = df_inputs, append = append)
