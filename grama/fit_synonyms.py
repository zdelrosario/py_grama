__all__ = [
    "fit_nls",
    "ft_nls",
]

## Collection of fitting synonyms: functions implemented in terms of other grama
## verbs

from grama import add_pipe, pipe, custom_formatwarning, df_make, eval_nls
from grama import Model, cp_function, cp_md_det
from toolz import curry


@curry
def fit_nls(
    df_data, md=None, verbose=True, **kwargs,
):
    r"""Fit a model with Nonlinear Least Squares (NLS)

    Estimate best-fit variable levels with nonlinear least squares (NLS), and
    return an executable model with those frozen best-fit levels.

    Note: This is a *synonym* for eval_nls(); see the documentation for
    eval_nls() for keyword argument options available beyond those listed here.

    Args:
        df_data (DataFrame): Data for estimating best-fit variable levels.
            Variables not found in df_data optimized for fitting.
        md (gr.Model): Model to analyze. All model variables
            selected for fitting must be bounded or random. Deterministic
            variables may have semi-infinite bounds.

    Returns:
        gr.Model: Model for evaluation with best-fit variables frozen to
            optimized levels.

    """
    ## Check invariants
    if md is None:
        raise ValueError("Must provide model md")

    ## Run eval_nls to fit model parameter values
    df_fit = eval_nls(md, df_data=df_data, append=True, **kwargs)
    ## Select best-fit values
    df_best = df_fit.sort_values(by="mse", axis=0).iloc[[0]]
    if verbose:
        print(df_best)

    ## Determine variables to fix
    var_fixed = list(set(md.var).intersection(set(df_best.columns)))
    var_remain = list(set(md.var).difference(set(var_fixed)))

    if len(var_remain) == 0:
        raise ValueError("Resulting model is constant!")

    ## Assemble and return fitted model
    if md.name is None:
        name = "(Fitted Model)"
    else:
        name = md.name + " (Fitted)"

    md_res = (
        Model(name)
        >> cp_function(
            lambda x: df_best[var_fixed].values,
            var=var_remain,
            out=var_fixed,
            name="Fix variable levels",
        )
        >> cp_md_det(md=md)
    )

    return md_res


ft_nls = add_pipe(fit_nls)
