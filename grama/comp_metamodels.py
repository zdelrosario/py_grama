__all__ = ["comp_metamodel", "cp_metamodel"]

## Fitting via statsmodels package
import grama as gr
from grama import pipe
from toolz import curry

## Fit a metamodel
# --------------------------------------------------
@curry
def comp_metamodel(model, n=1, ev=None, ft=None, seed=None):
    r"""Create a metamodel

    Composition: Create a metamodel from an existing model. This convenience
    function essentially applies a recipe of Evaluation followed by Fitting.
    Default methods are Latin Hypercube Evaluation and Ordinary Least Squares
    Fitting with linear features.

    Args:
        model (gr.model): Original model, to be evaluated and fit
        n (numeric): Number of samples to draw
        ev (gr.eval_): Evaluation strategy, default eval_lhs
        ft (gr.fit_): Fitting strategy, default fit_ols w/ linear features
        seed (int): Random seed, default None

    Returns:
        gr.model: Metamodel

    """
    ## Extract model information
    inputs = model.domain.inputs
    outputs = model.outputs

    ## Assign default arguments
    if ev is None:
        ev = gr.eval_lhs

    if ft is None:
        # Linear features for each output
        sum_inputs = "+".join(inputs)
        formulae = list(map(lambda output: output + "~" + sum_inputs, outputs))

        ft = lambda df: gr.fit_ols(
            df, formulae=formulae, domain=model.domain, density=model.density
        )

    ## Generate data
    df_results = ev(model, n_samples=n, seed=seed)

    ## Fit a model
    model = ft(df_results)

    return model


@pipe
def cp_metamodel(*args, **kwargs):
    return comp_metamodel(*args, **kwargs)
