__all__ = ["eval_lhs", "ev_lhs"]

try:
    from pyDOE import lhs

except ModuleNotFoundError:
    pass

import warnings
from grama import add_pipe, eval_df, custom_formatwarning, pipe
from scipy.stats import norm, lognorm
from numbers import Integral
from numpy import tile, linspace, zeros, isfinite
from numpy.linalg import cholesky, inv
from numpy.random import random
from numpy.random import seed as set_seed
from pandas import DataFrame
from toolz import curry


## Latin Hypercube Sampling (LHS)
# --------------------------------------------------
@curry
def eval_lhs(
    model,
    n=None,
    df_det=None,
    seed=None,
    append=True,
    skip=False,
    criterion=None,
):
    r"""Latin Hypercube evaluation
    Evaluates a given model on a latin hypercube sample (LHS) using the model's
    density.
    Args:
        model (gr.Model): Model to evaluate
        n (numeric): Number of LHS samples to draw
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        seed (int): Random seed to use
        append (bool): Append results to conservative inputs?
        skip (bool): Skip evaluation of the functions?
        criterion (str): flag for LHS sample criterion
            allowable values: None, "center" ("c"), "maxmin" ("m"),
            "centermaxmin" ("cm"), "correlation" ("corr")
    Returns:
        DataFrame: Results of evaluation or unevaluated design
    Notes:
        - Wrapper on pyDOE.lhs
    """
    ## Set seed only if given
    if seed is not None:
        set_seed(seed)

    ## Ensure sample count is int
    if not isinstance(n, Integral):
        print("eval_lhs() is rounding n...")
        n = int(n)

    ## Draw samples
    try:
        df_quant = DataFrame(
            data=lhs(model.n_var_rand, samples=n), columns=model.var_rand
        )
    except NameError as e:
        error_string = str(e)
        raise NameError(
            error_string
            + "\n\nThis function requires the `pyDOE` package. "
            + "Try running the following to install the package:\n"
            "    pip install pyDOE"
        )

    ## Convert samples to desired marginals
    df_rand = model.density.pr2sample(df_quant)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        return df_samp
    return eval_df(model, df=df_samp, append=append)


ev_lhs = add_pipe(eval_lhs)
