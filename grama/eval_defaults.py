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

import grama as gr
from grama import pipe
from toolz import curry

## Default evaluation function
# --------------------------------------------------
@curry
def eval_df(model, df=None, append=True):
    """Evaluate model at given values

    Evaluates a given model at a given dataframe.

    Args:
        model (gr.Model): Model to evaluate
        df (DataFrame): Input dataframe to evaluate
        append (bool): Append results to original dataframe?

    Returns:
        DataFrame: Results of model evaluation

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
    """Evaluate model at nominal values

    Evaluates a given model at a model nominal conditions (median).

    Args:
        model (gr.Model): Model to evaluate
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        append (bool): Append results to nominal inputs?
        skip (bool): Skip evaluation?

    Returns:
        DataFrame: Results of nominal model evaluation or unevaluated design

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
    """Finite-difference gradient approximation

    Evaluates a given model with a central-difference stencil to approximate the
    gradient.

    Args:
        model (gr.Model): Model to differentiate
        h (numeric): finite difference stepsize,
            single (scalar): or per-input (np.array)
        df_base (DataFrame): Base-points for gradient calculations
        varsdiff (list(str)): list of variables to differentiate
            NOT IMPLEMENTED
        append (bool): Append results to base point inputs?
        skip (bool): Skip evaluation?

    Returns:
        DataFrame: Gradient approximation or unevaluated design

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

    outputs = model.out
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
                columns=model.var,
                data=-stencil + df_base[model.var].iloc[[row_i]].values
            ),
            append = False
        )

        df_right = eval_df(
            model,
            pd.DataFrame(
                columns=model.var,
                data=+stencil + df_base[model.var].iloc[[row_i]].values
            ),
            append = False
        )

        res = (stepscale * (df_right[outputs] - df_left[outputs]).values).flatten()

        df_grad = pd.DataFrame(
            columns=grad_labels,
            data=[res]
        )

        results.append(df_grad)

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

    Args:
        model (gr.Model): Model to evaluate
        quantiles (numeric): lower quantile value(s) for conservative
            evaluation; can be single value for all inputs, array
            of values for each random variable, or None for default 0.01.
            values in [0, 0.5]
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        append (bool): Append results to conservative inputs?
        skip (bool): Skip evaluation?

    Returns:
        DataFrame: Conservative evaluation or unevaluated design

    """
    ## Default behavior
    if quantiles is None:
        print("eval_conservative() using quantile default 0.01;")
        print("provide `quantiles` keyword for non-default behavior.")
        quantiles = [0.01] * model.n_var_rand

    ## Handle scalar vs vector quantiles
    try:
        len(quantiles)
    except TypeError:
        quantiles = [quantiles] * model.n_var_rand

    ## Modify quantiles for conservative directions
    quantiles = [
        0.5 + (0.5 - quantiles[i]) * model.density.marginals[
            model.var_rand[i]
        ].sign \
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
