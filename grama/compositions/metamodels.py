## Fitting via statsmodels package
from .. import core
from .. import evals
from .. import fitting
from toolz import curry

## Fit a metamodel
@curry
def cp_metamodel(model, n=1, ev=None, ft=None, seed=None):
    """Create a metamodel

    @param model Original model, to be evaluated and fit
    @param n Number of samples to draw
    @param ev Evaluation strategy, default ev_lhs
    @param ft Fitting strategy, default ft_ols w/ linear features
    @param seed Random seed, default None
    """
    ## Extract model information
    inputs  = model.domain.inputs
    outputs = model.outputs

    ## Assign default arguments
    if ev is None:
        ev = evals.ev_lhs

    if ft is None:
        # Linear features for each output
        sum_inputs = "+".join(inputs)
        formulae = list(map(
            lambda output: output + "~" + sum_inputs,
            outputs
        ))

        ft = lambda df: fitting.ft_ols(
            df,
            formulae=formulae,
            domain=model.domain,
            density=model.density
        )

    ## Generate data
    df_results = ev(model, n_samples=n, seed=seed)

    ## Fit a model
    model = ft(df_results)

    return model
