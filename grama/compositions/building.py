__all__ = [
    "comp_function",
    "cp_function"
]

from .. import core
from .. import evals
from .. import fitting
from ..tools import pipe
from toolz import curry

## Model Building Interface (MBI) tools
##################################################
@curry
def comp_function(
    model, fun=None, var=None, out=None, name=None
):
    """Add a function to a model.

    @param model [gr.model] Model to compose
    @param fun [function] Function taking R^d -> R^r
    @param var [list(string) or int] List of variable names or number of inputs
    @param out [list(string) or int] List of output names or number of outputs

    @returns [gr.model] New model with added function

    @pre (len(var) == d) | (var == d)
    @pre (len(out) == r) | (var == r)
    """
    model_new = model.copy()

    # Check inputs
    if fun is None:
        raise ValueError("`fun` must be a valid function")

    if name is None:
        name = fun.__name__

    # Create variable names, if necessary
    if isinstance(var, int):
        i0 = model_new.n_var
        i1 = model_new.n_var + var
        var = ["x{}".format(i) for i in range(i0, i1)]
    elif var is None:
        raise ValueError("`var` must be list or int")

    # Create output names, if necessary
    if isinstance(out, int):
        i0 = model_new.n_out
        i1 = model_new.n_out + out
        out = ["y{}".format(i) for i in range(i0, i1)]
    elif out is None:
        raise ValueError("`out` must be list or int")

    ## Add new function
    model_new.functions.append(
        core.Function(fun, var, out, name)
    )
    model_new.update()

    return model_new

@pipe
def cp_function(*args, **kwargs):
    comp_function(*args, **kwargs)
