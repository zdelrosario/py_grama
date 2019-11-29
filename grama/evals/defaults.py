__all__ = [
    "eval_nominal",
    "ev_nominal",
    "eval_grad_fd",
    "ev_grad_fd",
    "eval_conservative",
    "ev_conservative"
]

import numpy as np
import pandas as pd
import itertools

from .. import core
from ..core import pipe
from toolz import curry

## Nominal evaluation
# --------------------------------------------------
@curry
def eval_nominal(model, append=True):
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

    return model >> core.ev_df(df=df_inputs, append=append)

@pipe
def ev_nominal(*args, **kwargs):
    return eval_nominal(*args, **kwargs)

## Gradient finite-difference evaluation
# --------------------------------------------------
@curry
def eval_grad_fd(model, df_base=None, append=True, h=1e-8):
    """Evaluates a given model with a central-difference stencil to
    approximate the gradient

    @param model Valid grama model
    @param df_base DataFrame of base-points for gradient calculations
    @param append bool flag append results to df_base (True) or
           return results in separate DataFrame (False)?
    @param h finite difference stepsize,
           single (scalar) or per-input (np.array)

    @pre (not isinstance(h, collections.Sequence)) | (h.shape[0] == model.n_in)
    """
    ## Build stencil
    stencil = np.eye(model.n_in) * h
    scaler  = np.tile(np.atleast_2d(0.5/h).T, (1, model.n_out))

    outputs = model.outputs
    inputs = model.domain.inputs
    nested_labels = [
        list(map(lambda s_out: "D" + s_out + "_D" + s_in, outputs)) for s_in in inputs
    ]
    grad_labels = list(itertools.chain.from_iterable(nested_labels))

    ## Loop over df_base
    results = [] # TODO: Preallocate?
    for row_i in range(df_base.shape[0]):
        df_left = core.eval_df(
            model,
            pd.DataFrame(
                columns = inputs,
                data = -stencil + df_base[inputs].iloc[[row_i]].values
            ),
            append = False
        )

        df_right = core.eval_df(
            model,
            pd.DataFrame(
                columns = inputs,
                data = +stencil + df_base[inputs].iloc[[row_i]].values
            ),
            append = False
        )

        res = (scaler * (df_right - df_left).values).flatten()

        df_grad = pd.DataFrame(
            columns = grad_labels,
            data = [res]
        )

        results.append(df_grad)

    ## TODO: append
    return pd.concat(results)

@pipe
def ev_grad_fd(*args, **kwargs):
    return eval_grad_fd(*args, **kwargs)

## Conservative quantile evaluation
# --------------------------------------------------
@curry
def eval_conservative(model, quantiles=None, append=True):
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

    return core.eval_df(model, df = df_inputs, append = append)

@pipe
def ev_conservative(*args, **kwargs):
    return eval_conservative(*args, **kwargs)
