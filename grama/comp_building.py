__all__ = [
    "comp_freeze",
    "cp_freeze",
    "comp_function",
    "cp_function",
    "comp_vec_function",
    "cp_vec_function",
    "comp_md_det",
    "cp_md_det",
    "comp_md_sample",
    "cp_md_sample",
    "comp_bounds",
    "cp_bounds",
    "comp_copula_independence",
    "cp_copula_independence",
    "comp_copula_gaussian",
    "cp_copula_gaussian",
    "comp_marginals",
    "cp_marginals",
    "getvars",
]

from grama import add_pipe, CopulaGaussian, CopulaIndependence, Density, \
    Function, FunctionModel, FunctionVectorized, Marginal, MarginalNamed, \
    pipe, tran_copula_corr
from .eval_defaults import eval_sample
from collections import ChainMap
from pandas import concat, DataFrame
from toolz import curry


## Model Building Interface (MBI) tools
##################################################
def _comp_function_data(model, fun, var, out, name, runtime):
    r"""Internal function builder
    """
    model_new = model.copy()

    # Check invariants
    if fun is None:
        raise ValueError("`fun` must be a valid function")

    if name is None:
        name = "f{}".format(len(model.functions))
    else:
        if name in [f.name for f in model.functions]:
            raise ValueError("`name` must be unique")

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

    # Check DAG invariants
    args_inter = set(out).intersection(set(var))
    if len(args_inter) > 0:
        raise ValueError(
            "`out` must not intersect `var`\n" + "intersection = {}".format(args_inter)
        )
    out_var_inter = set(out).intersection(set(model.var))
    if len(out_var_inter) > 0:
        raise ValueError(
            "`out` must not intersect model.var"
            + "intersection = {}".format(out_var_inter)
        )
    out_out_inter = set(out).intersection(set(model.out))
    if len(out_out_inter) > 0:
        raise ValueError(
            "`out` must not intersect model.out"
            + "intersection = {}".format(out_out_inter)
        )

    return fun, var, out, name, runtime


# Extract function's variable names
# -------------------------
def getvars(f):
    r"""Get a function's variable names

    Convenience function for extracting a function's variable names. Intended for use with gr.cp_function().

    Args:
        f (function): Function whose variable names are desired

    Returns:
        tuple: Variable names

    Examples::
        import grama as gr

        def fun(x, y, z):
            return x + y + z

        md = (
            gr.Model("Test model")
            >> gr.cp_function(
                fun=fun,
                var=gr.getvars(fun),
                out=["w"],
            )
        )
    """
    return f.__code__.co_varnames

# Freeze inputs
# -------------------------
@curry
def comp_freeze(model, df=None, **var):
    r"""Freeze inputs to a model

    Composition. Remove inputs from a model by "freezing" them to fixed values.

    Args:
        model (gr.Model): Model to compose
        df (pd.DataFrame): DataFrame of values for freeze
        var (dict): Dictionary of inputs to freeze (keys) to specific values (value)
            Provide each key/value pair as a keyword argument

    Returns:
        gr.Model: New model with frozen inputs

    Examples::
        import grama as gr

    """
    ## Check invariants
    # Process DataFrame if provided
    if not df is None:
        if df.shape[0] > 1:
            raise ValueError(
                "Provided DataFrame must have only one row."
            )
        var = dict(zip(df.columns, df.values.flatten()))

    # All variables are provided
    var_miss = set(set(var.keys())).difference(model.var)
    if len(var_miss) != 0:
        raise ValueError(
            "All inputs listed in `var` argument must be present in model.var.\n" +
            "Missing inputs {}".format(var_miss)
        )

    if any(map(lambda x: hasattr(x, "__iter__"), var.values())):
        raise ValueError(
            "Iterable input value detected. All values provided must be scalars."
        )

    ## Create "freezer function"
    var_diff = list(set(set(model.var)).difference(var.keys()))
    f = FunctionVectorized(
        lambda df: df.assign(**var),
        var_diff,
        list(var.keys()),
        "(Freeze inputs: {})".format(list(var.keys())),
        0
    )

    ## Add to model
    model_new = model.copy()
    # Remove bounds, if they exist
    for k in var.keys():
        model_new.domain.bounds.pop(k, None)
    # Remove marginals, if they exist
    for k in var.keys():
        model_new.density.marginals.pop(k, None)
    # Reset copula
    model_new.density = Density(
        marginals=model_new.density.marginals,
        copula=CopulaIndependence(model_new.var_rand),
    )
    print("... comp_freeze() is resetting copula to independence copula")

    # Add new functions
    model_new.functions.insert(0, f)
    # Update before return
    model_new.update()

    return model_new


cp_freeze = add_pipe(comp_freeze)


# Add a lambda function
# -------------------------
@curry
def comp_function(model, fun=None, var=None, out=None, name=None, runtime=0):
    r"""Add a function to a model

    Composition. Add a (non-vectorized) function to an existing model. See ``gr.comp_vec_function()`` to add a function that is vectorized over DataFrames.

    Args:
        model (gr.Model): Model to compose
        fun (function): Function taking at least one real input and returns at least one real output (fun: R^d -> R^r). Each input to `fun` must be a scalar (See examples below).
        var (list(string)): List of variable names or number of inputs
        out (list(string)): List of output names or number of outputs
        runtime (numeric): Estimated single-eval runtime (in seconds)

    Returns:
        gr.Model: New model with added function

    @pre (len(var) == d) | (var == d)
    @pre (len(out) == r) | (var == r)

    Examples::

        import grama as gr
        ## Simple example
        md = (
            gr.Model("test")
            >> gr.cp_function(
                fun=lambda x: x,
                var=["x"],
                out=["y"],
                name="identity"
            )
        )

        ## Providing a function with multiple inputs
        md2 = (
            gr.Model("test 2")
            >> gr.cp_function(
                fun=lambda x, y: x + y,
                var=["x", "y"],
                out=["f"],
            )
        )

        ## Providing a function with multiple inputs and multiple outputs
        md3 = (
            gr.Model("test 3")
            >> gr.cp_function(
                fun=lambda x, y: [x + y, x - y],
                var=["x", "y"],
                out=["f", "g"],
            )
        )

    """
    model_new = model.copy()

    ## Dispatch to core builder for consistent behavior
    fun, var, out, name, runtime = _comp_function_data(
        model, fun, var, out, name, runtime
    )

    ## Add new function
    model_new.functions.append(Function(fun, var, out, name, runtime))

    model_new.update()
    return model_new


cp_function = add_pipe(comp_function)

# Add vectorized function
# -------------------------
@curry
def comp_vec_function(model, fun=None, var=None, out=None, name=None, runtime=0):
    r"""Add a vectorized function to a model

    Composition. Add a function to an existing model. Function must be
    vectorized over DataFrames, and must add new columns matching its `out`
    set. See ``gr.cp_function()`` to add a non-vectorized function.

    Notes:
        The helper function ``gr.df_make()`` is useful for constructing a vectorized lambda function (see Examples below).

    Args:
        model (gr.model): Model to compose
        fun (function): Function taking R^d -> R^r; must be *vectorized* over DataFrames; it must take a DataFrame as input and return a new DataFrame
        var (list(string)): List of variable names or number of inputs
        out (list(string)): List of output names or number of outputs
        runtime (numeric): Estimated single-eval runtime (in seconds)

    Returns:
        gr.model: New model with added function

    @pre (len(var) == d) | (var == d)
    @pre (len(out) == r) | (var == r)

    Examples::

        import grama as gr
        ## Simple example
        md = (
            gr.Model("Test")
            >> gr.cp_vec_function(
                fun=lambda df: gr.df_make(y=1 + 0.5 * df.x),
                var=["x"],
                out=["y"],
                name="Simple linear function",
            )
        )

    """
    model_new = model.copy()

    ## Dispatch to core builder for consistent behavior
    fun, var, out, name, runtime = _comp_function_data(
        model, fun, var, out, name, runtime
    )

    ## Add new vectorized function
    model_new.functions.append(FunctionVectorized(fun, var, out, name, runtime))

    model_new.update()
    return model_new


cp_vec_function = add_pipe(comp_vec_function)

# Add model as deterministic function
# -------------------------
@curry
def comp_md_det(model, md=None):
    r"""Add a Model with deterministic evaluation

    Composition. Add a model as function to an existing model. Evaluate the
    model deterministically (ignore any random variables).

    Args:
        model (gr.model): Model to compose
        md (gr.model): Model to add as function

    Returns:
        gr.model: New model with added function

    Examples::

        import grama as gr
        from grama.models import make_cantilever_beam
        ## Use functions from beam model, but introduce new marginals
        md_plate = (
            gr.Model("New beam model")
            >> gr.cp_md_det(md=make_cantilever_beam())
            >> gr.cp_marginals(
                H=gr.marg_mom("norm", mean=1000, cov=0.1),
                V=gr.marg_mom("norm", mean=500, cov=0.1),
            )
            >> gr.cp_copula_independence()
        )

    """
    if md is None:
        raise ValueError("Must provide `md` argument")

    model_new = model.copy()
    model_new.functions.append(FunctionModel(md))

    model_new.update()
    return model_new


cp_md_det = add_pipe(comp_md_det)

# Add model as sampled function
# -------------------------
@curry
def comp_md_sample(model, md=None, param=None, rand2out=False):
    r"""Add a Model with sampled evaluation

    Composition. Add a model as function to an existing model. Evaluate the
    model via sampling (one sample per evaluation). Use `param` to turn model
    parameters into variables of new model. Random variables of composed model
    are turned into outputs of new model.

    Args:
        model (gr.model): Model to compose
        md (gr.model): Model to add as function
        param (dict): Parameters in md to treat as var; entries must be
            of the form "var": ("param1", "param2", ...)
        rand2out (bool): Add model's var_rand to outputs (to track values)

    Returns:
        gr.model: New model with added function

    Examples::

    """
    ## Check invariants
    if md is None:
        raise ValueError("Must provide `md` argument")

    diff = set(param.keys()).difference(set(md.var_rand))
    if not len(diff) == 0:
        raise ValueError(
            "param must be in md.var_rand;\n"
            "{{param}} - {{md.var_rand}} = {}".format(diff)
        )

    ## Setup
    model_new = model.copy()
    md_new = md.copy()

    ## Construct parameter mapping
    if param is None:
        param_dict = {}
    else:
        param_dict = dict(
            ChainMap(
                *[
                    {key + "_" + v: (key, v) for v in values}
                    for key, values in param.items()
                ]
            )
        )

    ## Compute new model var + out
    if rand2out:
        out = list(md.out) + list(md.var_rand)
    else:
        out = list(md.out)
    var = list(md.var_det) + list(param_dict.keys())

    ## Construct evaluator
    def _ev(md, df):
        df_res = DataFrame()

        for i in range(df.shape[0]):
            ## Edit model
            for var, pair in param_dict.items():
                md.density.marginals[pair[0]].d_param[pair[1]] = df.iloc[i][var]

            ## Evaluate
            df_tmp = eval_sample(
                md, n=1, df_det=df.iloc[[i]][list(md.var_det)].reset_index(drop=True)
            )

            ## Concatenate
            df_res = concat((df_res, df_tmp), axis=0)

        return df_res[out].reset_index(drop=True)

    ## Construct FunctionModel and assign
    model_new.functions.append(FunctionModel(md_new, ev=_ev, var=var, out=out))

    model_new.update()
    return model_new


cp_md_sample = add_pipe(comp_md_sample)

# Add bounds
# -------------------------
@curry
def comp_bounds(model, **kwargs):
    r"""Add variable bounds to a model

    Composition. Add variable bounds to an existing model. Bounds are specified by iterable; the model variable name is specified by the keyword argument name.

    Args:
        model (gr.model): Model to modify

    Kwargs:
        var (iterable): Bound information; keyword argument name is targeted variable, value should be a length 2 iterable of the form (lower_bound, upper_bound)

    Returns:
        gr.model: Model with new marginals

    @pre len(var) >= 2

    Examples::

        import grama as gr
        md = (
            gr.Model("Simple Model")
            >> gr.cp_function(
                lambda x: x[0] + x[1],
                var=["x0", "x1"],
                out=1
            )
            >> gr.cp_bounds(
                x0=(-1, 1),    # Finite bounds
                x1=(0, np.inf) # Semi-infinite bounds
            )
        )

    """
    new_model = model.copy()

    ## Parse keyword arguments
    for key, value in kwargs.items():
        ## Add new bound
        new_model.domain.bounds[key] = [value[0], value[1]]

    new_model.update()
    return new_model


cp_bounds = add_pipe(comp_bounds)

# Add marginals
# -------------------------
@curry
def comp_marginals(model, **kwargs):
    r"""Add marginals to a model

    Composition. Add marginals to an existing model. Marginals are specified
    either by dictionary entries or by gr.Marginal() object. The model variable
    name is specified by the keyword argument name.

    Notes:
        Several helper functions are available to fit marginal distributions

        - ``gr.marg_fit()`` fits a distribution using a dataset (via maximum likelihood estimation)
        - ``gr.marg_mom()`` fits a distribution using moments (via the method of moments)
        - ``gr.marg_gkde()`` fits a gaussian kernel density using a dataset

    Args:
        model (gr.model): Model to modify
        var (dict OR gr.Marginal): Marginal information

    Returns:
        gr.model: Model with new marginals

    Examples::

        import grama as gr
        ## Print all of the grama-supported distributions
        print(gr.valid_dist.keys())
        ## Construct a simple example model
        md = (
            gr.Model()
            >> gr.cp_function(
                lambda x: x[0] + x[1],
                var=["x0", "x1"],
                out=["y"],
            )
            >> gr.cp_marginals(
                x0=gr.marg_mom("norm", mean=0, sd=1),
            )
        )

    """
    new_model = model.copy()

    ## Parse keyword arguments
    for key, value in kwargs.items():
        value_copy = value.copy()

        ## Handle dictionary input
        if isinstance(value_copy, dict):
            ## Check for named marginal
            try:
                dist = value_copy.pop("dist")
            except KeyError:
                raise NotImplementedError(
                    "Must give distribution name when using dict input"
                )

            try:
                sign = value_copy.pop("sign")
            except KeyError:
                sign = 0

            new_model.density.marginals[key] = MarginalNamed(
                sign=sign, d_name=dist, d_param=value_copy
            )

        ## Handle Marginal input
        if isinstance(value_copy, Marginal):
            new_model.density.marginals[key] = value_copy

    new_model.update()
    return new_model


cp_marginals = add_pipe(comp_marginals)

# Add copula
##################################################
@curry
def comp_copula_independence(model):
    r"""Add an independence copula to model

    Composition. Add an independence copula to an existing model.

    NOTE: Independence of random variables is a *very* strong assumption! Recommend using comp_copula_gaussian instead.

    Args:
        model (gr.model): Model to modify

    Returns:
        gr.model: Model with independence copula

    Examples::

        import grama as gr
        md = (
            gr.Model()
            >> gr.cp_marginals(
                x0=gr.marg_mom("norm", mean=0, sd=1),
                x1=gr.marg_mom("beta", mean=0, sd=1, skew=0, kurt=2),
            )
            >> gr.cp_copula_independence()
        )

    """
    new_model = model.copy()
    new_model.density = Density(
        marginals=model.density.marginals,
        copula=CopulaIndependence(new_model.var_rand),
    )
    new_model.update()

    return new_model


cp_copula_independence = add_pipe(comp_copula_independence)

# -------------------------
@curry
def comp_copula_gaussian(model, df_corr=None, df_data=None):
    r"""Add a Gaussian copula to model

    Composition. Add a gaussian copula to an existing model.

    Args:
        model (gr.model): Model to modify
        df_corr (DataFrame): Correlation information
        df_data (DataFrame): Data for automated fitting

    Returns:
        gr.model: Model with Gaussian copula

    Examples::

        import grama as gr
        ## Manual assignment
        md_manual = (gr.Model()
            >> gr.cp_marginals(
                x0=gr.marg_mom("norm", mean=0, sd=1),
                x1=gr.marg_mom("uniform", mean=0, sd=1),
            )
            >> gr.cp_copula_gaussian(
                # Specify correlation structure explicitly
                df_corr=gr.df_make(var1="x0", var2="x1", corr=0.5)
            )
        )
        ## Automated fitting
        from grama.data import df_stang
        md_auto = (
            gr.Model()
            >> gr.cp_marginals(
                E=gr.marg_fit("norm", df_stang.E),
                mu=gr.marg_fit("beta", df_stang.mu),
                thick=gr.marg_fit("norm", df_stang.thick)
            )
            >> gr.cp_copula_gaussian(df_data=df_stang)
        )

    """
    if not (df_corr is None):
        new_model = model.copy()
        new_model.density = Density(
            marginals=model.density.marginals,
            copula=CopulaGaussian(list(model.density.marginals.keys()), df_corr,),
        )
        new_model.update()

        return new_model

    if not (df_data is None):
        new_model = model.copy()
        df_corr = tran_copula_corr(df_data, model=new_model)

        new_model.density = Density(
            marginals=model.density.marginals,
            copula=CopulaGaussian(list(model.density.marginals.keys()), df_corr,),
        )
        new_model.update()

        return new_model

    else:
        raise ValueError("Must provide df_corr or df_data")


cp_copula_gaussian = add_pipe(comp_copula_gaussian)
