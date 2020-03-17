__all__ = [
    "eval_df",
    "ev_df",
    "eval_nominal",
    "ev_nominal",
    "eval_grad_fd",
    "ev_grad_fd",
    "eval_conservative",
    "ev_conservative",
]

from numpy import ones, eye, tile, atleast_2d
from pandas import DataFrame, concat
import itertools

import grama as gr
from grama import pipe
from toolz import curry

## Default evaluation function
# --------------------------------------------------
@curry
def eval_df(model, df=None, append=True):
    r"""Evaluate model at given values

    Evaluates a given model at a given dataframe.

    Args:
        model (gr.Model): Model to evaluate
        df (DataFrame): Input dataframe to evaluate
        append (bool): Append results to original dataframe?

    Returns:
        DataFrame: Results of model evaluation

    Examples:

        >>> import grama as gr
        >>> from grama.models import make_test
        >>> md = make_test()
        >>> df = gr.df_make(x0=0, x1=1, x2=2)
        >>> md >> gr.ev_df(df=df)

    """
    if df is None:
        raise ValueError("No input df given")
    if len(model.functions) == 0:
        raise ValueError("Given model has no functions")

    df_res = model.evaluate_df(df)

    if append:
        df_res = concat([df.reset_index(drop=True), df_res], axis=1)

    return df_res


@pipe
def ev_df(*args, **kwargs):
    return eval_df(*args, **kwargs)


## Nominal evaluation
# --------------------------------------------------
@curry
def eval_nominal(model, df_det=None, append=True, skip=False):
    r"""Evaluate model at nominal values

    Evaluates a given model at a model nominal conditions (median).

    Args:
        model (gr.Model): Model to evaluate
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        append (bool): Append results to nominal inputs?
        skip (bool): Skip evaluation of the functions?

    Returns:
        DataFrame: Results of nominal model evaluation or unevaluated design

    Examples:

        >>> import grama as gr
        >>> from grama.models import make_test
        >>> md = make_test()
        >>> md >> gr.ev_nominal(df_det="nom")

    """
    ## Draw from underlying gaussian
    quantiles = ones((1, model.n_var_rand)) * 0.5  # Median

    ## Convert samples to desired marginals
    df_pr = DataFrame(data=quantiles, columns=model.var_rand)
    ## Convert samples to desired marginals
    df_rand = model.density.pr2sample(df_pr)
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
def eval_grad_fd(model, h=1e-8, df_base=None, var=None, append=True, skip=False):
    r"""Finite-difference gradient approximation

    Evaluates a given model with a central-difference stencil to approximate the
    gradient.

    Args:
        model (gr.Model): Model to differentiate
        h (numeric): finite difference stepsize,
            single (scalar): or per-input (array)
        df_base (DataFrame): Base-points for gradient calculations
        var (list(str) or string): list of variables to differentiate,
            or flag; "rand" for var_rand, "det" for var_det
        append (bool): Append results to base point inputs?
        skip (bool): Skip evaluation of the functions?

    Returns:
        DataFrame: Gradient approximation or unevaluated design

    @pre (not isinstance(h, collections.Sequence)) |
         (h.shape[0] == df_base.shape[1])

    Examples:

        >>> import grama as gr
        >>> from grama.models import make_cantilever_beam
        >>> md = make_cantilever_beam()
        >>> df_nom = md >> gr.ev_nominal(df_det="nom")
        >>> df_grad = md >> gr.ev_grad_fd(df_base=df_nom)
        >>> df_grad >> gr.tf_gather("var", "val", gr.everything())

    """
    ## Check invariants
    if not set(model.var).issubset(set(df_base.columns)):
        raise ValueError("model.var must be subset of df_base.columns")
    if var is None:
        var = model.var
    elif isinstance(var, str):
        if var == "rand":
            var = model.var_rand
        elif var == "det":
            var = model.var_det
        else:
            raise ValueError("var flag not recognized; use 'rand' or 'det'")
    else:
        if not set(var).issubset(set(model.var)):
            raise ValueError("var must be subset of model.var")
    var_fix = list(set(model.var).difference(set(var)))

    ## TODO
    if skip == True:
        raise NotImplementedError("skip not implemented")

    ## Build stencil
    n_var = len(var)
    stencil = eye(n_var) * h
    stepscale = tile(atleast_2d(0.5 / h).T, (1, model.n_out))

    outputs = model.out
    nested_labels = [
        list(map(lambda s_out: "D" + s_out + "_D" + s_var, outputs)) for s_var in var
    ]
    grad_labels = list(itertools.chain.from_iterable(nested_labels))

    ## Loop over df_base
    results = []  # TODO: Preallocate?
    for row_i in range(df_base.shape[0]):
        ## Evaluate
        df_left = eval_df(
            model,
            gr.tran_outer(
                DataFrame(
                    columns=var, data=-stencil + df_base[var].iloc[[row_i]].values
                ),
                df_base[var_fix].iloc[[row_i]],
            ),
            append=False,
        )

        df_right = eval_df(
            model,
            gr.tran_outer(
                DataFrame(
                    columns=var, data=+stencil + df_base[var].iloc[[row_i]].values
                ),
                df_base[var_fix].iloc[[row_i]],
            ),
            append=False,
        )

        ## Compute differences
        res = (stepscale * (df_right[outputs] - df_left[outputs]).values).flatten()
        df_grad = DataFrame(columns=grad_labels, data=[res])

        results.append(df_grad)

    return concat(results).reset_index(drop=True)


@pipe
def ev_grad_fd(*args, **kwargs):
    return eval_grad_fd(*args, **kwargs)


## Conservative quantile evaluation
# --------------------------------------------------
@curry
def eval_conservative(model, quantiles=None, df_det=None, append=True, skip=False):
    r"""Evaluates a given model at conservative input quantiles

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
        skip (bool): Skip evaluation of the functions?

    Returns:
        DataFrame: Conservative evaluation or unevaluated design

    Examples:

        >>> import grama as gr
        >>> from grama.models import make_plate_buckle
        >>> md = make_plate_buckle()
        >>> md >> gr.ev_conservative(df_det="nom")

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
        0.5 + (0.5 - quantiles[i]) * model.density.marginals[model.var_rand[i]].sign
        for i in range(model.n_var_rand)
    ]
    quantiles = atleast_2d(quantiles)

    ## Draw samples
    df_pr = DataFrame(data=quantiles, columns=model.var_rand)
    ## Convert samples to desired marginals
    df_rand = model.density.pr2sample(df_pr)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        return df_samp
    else:
        return eval_df(model, df=df_samp, append=append)


@pipe
def ev_conservative(*args, **kwargs):
    return eval_conservative(*args, **kwargs)
