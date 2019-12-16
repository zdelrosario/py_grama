__all__ = [
    "eval_df",
    "ev_df",
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
from ..tools import pipe
from toolz import curry

## Default evaluation function
# --------------------------------------------------
@curry
def eval_df(model, df=None, append=True):
    """Evaluates a given model at a given dataframe

    @param df input dataframe to evaluate (Pandas.DataFrame)
    @param append bool flag; append results to original dataframe?
    """

    if df is None:
        raise ValueError("No input df given!")

    df_res = model.evaluate_df(df)

    if append:
        df_res = pd.concat([df.reset_index(drop=True), df_res], axis=1)

    return df_res

@pipe
def ev_df(*args, **kwargs):
    return eval_df(*args, **kwargs)

## Nominal evaluation
# --------------------------------------------------
@curry
def eval_nominal(model, df_det=None, append=True, skip=False):
    """Evaluates a given model at a model nominal conditions (median)

    @param append bool flag; append results to nominal inputs?

    Only implemented for gaussian copula distributions for now.
    """
    ## Draw from underlying gaussian
    quantiles = np.ones((1, model.n_var_rand)) * 0.5 # Median

    ## Convert samples to desired marginals
    df_quant = pd.DataFrame(data=quantiles, columns=model.var_rand)
    ## Convert samples to desired marginals
    df_rand = model.var_rand_quantile(df_quant)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        return df_samp
    else:
        return eval_df(model, df=df_samp, append=append)

@pipe
def ev_nominal(*args, **kwargs):
    return eval_nominal(*args, **kwargs)

## Gradient finite-difference evaluation
# --------------------------------------------------
@curry
def eval_grad_fd(
        model,
        h=1e-8,
        df_base=None,
        varsdiff=None,
        append=True,
        skip=False
):
    """Evaluates a given model with a central-difference stencil to
    approximate the gradient

    @param model Valid grama model
    @param h finite difference stepsize,
           single (scalar) or per-input (np.array)
    @param df_base DataFrame of base-points for gradient calculations
    @param varsdiff list of variables to differentiate TODO
    @param append bool flag append results to df_base (True) or
           return results in separate DataFrame (False)?
    @param skip

    @pre (not isinstance(h, collections.Sequence)) |
         (h.shape[0] == df_base.shape[1])
    """
    ## TODO
    if not (varsdiff is None):
        raise NotImplementedError("varsdiff non-default not implemented")
    ## TODO
    if skip == True:
        raise NotImplementedError("skip not implemented")

    ## Build stencil
    stencil = np.eye(model.n_var) * h
    stepscale = np.tile(np.atleast_2d(0.5/h).T, (1, model.n_out))

    outputs = model.outputs
    nested_labels = [
        list(map(lambda s_out: "D" + s_out + "_D" + s_var, outputs)) \
        for s_var in model.var
    ]
    grad_labels = list(itertools.chain.from_iterable(nested_labels))

    ## Loop over df_base
    results = [] # TODO: Preallocate?
    for row_i in range(df_base.shape[0]):
        df_left = eval_df(
            model,
            pd.DataFrame(
                columns = model.var,
                data = -stencil + df_base[model.var].iloc[[row_i]].values
            ),
            append = False
        )

        df_right = eval_df(
            model,
            pd.DataFrame(
                columns = model.var,
                data = +stencil + df_base[model.var].iloc[[row_i]].values
            ),
            append = False
        )

        res = (stepscale * (df_right[outputs] - df_left[outputs]).values).flatten()

        df_grad = pd.DataFrame(
            columns=grad_labels,
            data=[res]
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
def eval_conservative(model, quantiles=None, df_det=None, append=True, skip=False):
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
    @param df_det DataFrame deterministic samples
    @param append bool flag; append results to nominal inputs?

    @pre (len(quantiles) == model.n_in) || (quantiles is None)
    """
    if quantiles is None:
        quantiles = [0.01] * model.n_var_rand

    ## Modify quantiles for conservative directions
    quantiles = [
        0.5 + (0.5 - quantiles[i]) * model.density.marginals[i].sign \
        for i in range(model.n_var_rand)
    ]
    quantiles = np.atleast_2d(quantiles)

    ## Draw samples
    df_quant = pd.DataFrame(data=quantiles, columns=model.var_rand)
    ## Convert samples to desired marginals
    df_rand = model.var_rand_quantile(df_quant)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        return df_samp
    else:
        return eval_df(model, df=df_samp, append=append)

@pipe
def ev_conservative(*args, **kwargs):
    return eval_conservative(*args, **kwargs)
