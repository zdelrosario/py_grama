__all__ = [
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
]

from collections import ChainMap
import grama as gr
from grama import pipe
from toolz import curry
from pandas import concat, DataFrame

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
    if len(set(out).intersection(set(model.var))) > 0:
        raise ValueError("`out` must not intersect model.var")
    if len(set(out).intersection(set(model.out))) > 0:
        raise ValueError("`out` must not intersect model.out")

    return fun, var, out, name, runtime


# Add a lambda function
# -------------------------
@curry
def comp_function(model, fun=None, var=None, out=None, name=None, runtime=0):
    r"""Add a function to a model

    Composition. Add a function to an existing model.

    Args:
        model (gr.model): Model to compose
        fun (function): Function taking R^d -> R^r
        var (list(string)): List of variable names or number of inputs
        out (list(string)): List of output names or number of outputs
        runtime (numeric): Estimated single-eval runtime (in seconds)

    Returns:
        gr.model: New model with added function

    @pre (len(var) == d) | (var == d)
    @pre (len(out) == r) | (var == r)

    Examples:

        >>> import grama as gr
        >>> md = gr.Model("test") >> \
        >>>     gr.function(
        >>>         fun=lambda x: x,
        >>>         var=1,
        >>>         out=["y"],
        >>>         name="identity"
        >>>     )

    """
    model_new = model.copy()

    ## Dispatch to core builder for consistent behavior
    fun, var, out, name, runtime = _comp_function_data(
        model, fun, var, out, name, runtime
    )

    ## Add new function
    model_new.functions.append(gr.Function(fun, var, out, name, runtime))

    model_new.update()
    return model_new


@pipe
def cp_function(*args, **kwargs):
    return comp_function(*args, **kwargs)


# Add vectorized function
# -------------------------
@curry
def comp_vec_function(model, fun=None, var=None, out=None, name=None, runtime=0):
    r"""Add a vectorized function to a model

    Composition. Add a function to an existing model. Function must be
    vectorized over DataFrames.

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

    Examples:

        >>> import grama as gr
        >>> md = gr.Model("test") >> \
        >>>     gr.function(
        >>>         fun=lambda x: x,
        >>>         var=1,
        >>>         out=["y"],
        >>>         name="identity"
        >>>     )

    """
    model_new = model.copy()

    ## Dispatch to core builder for consistent behavior
    fun, var, out, name, runtime = _comp_function_data(
        model, fun, var, out, name, runtime
    )

    ## Add new vectorized function
    model_new.functions.append(gr.FunctionVectorized(fun, var, out, name, runtime))

    model_new.update()
    return model_new


@pipe
def cp_vec_function(*args, **kwargs):
    return comp_vec_function(*args, **kwargs)


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

    Examples:

    """
    if md is None:
        raise ValueError("Must provide `md` argument")

    model_new = model.copy()
    model_new.functions.append(gr.FunctionModel(md))

    model_new.update()
    return model_new


@pipe
def cp_md_det(*args, **kwargs):
    return comp_md_det(*args, **kwargs)


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

    Examples:

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
            df_tmp = gr.eval_monte_carlo(
                md, n=1, df_det=df.iloc[[i]][list(md.var_det)].reset_index(drop=True)
            )

            ## Concatenate
            df_res = concat((df_res, df_tmp), axis=0)

        return df_res[out].reset_index(drop=True)

    ## Construct FunctionModel and assign
    model_new.functions.append(gr.FunctionModel(md_new, ev=_ev, var=var, out=out))

    model_new.update()
    return model_new


@pipe
def cp_md_sample(*args, **kwargs):
    return comp_md_sample(*args, **kwargs)


# Add bounds
# -------------------------
@curry
def comp_bounds(model, **kwargs):
    r"""Add variable bounds to a model

    Composition. Add variable bounds to an existing model. Bounds are specified
    by iterable; the model variable name is specified by the keyword argument
    name.

    Args:
        model (gr.model): Model to modify
        var (iterable): Bound information

    Returns:
        gr.model: Model with new marginals

    @pre len(var) >= 2

    Examples:

        >>> import grama as gr
        >>> md = gr.Model() >> \
        >>>     cp_function(
        >>>         lambda x: x[0] + x[1],
        >>>         var=["x0", "x1"],
        >>>         out=1
        >>>     ) >> \
        >>>     cp_bounds(
        >>>         x0=(-1, 1),
        >>>         x1=(0, np.inf)
        >>>     )

    """
    new_model = model.copy()

    ## Parse keyword arguments
    for key, value in kwargs.items():
        ## Add new bound
        new_model.domain.bounds[key] = [value[0], value[1]]

    new_model.update()
    return new_model


@pipe
def cp_bounds(*args, **kwargs):
    return comp_bounds(*args, **kwargs)


# Add marginals
# -------------------------
@curry
def comp_marginals(model, **kwargs):
    r"""Add marginals to a model

    Composition. Add marginals to an existing model. Marginals are specified
    either by dictionary entries or by gr.Marginal() object. The model variable
    name is specified by the keyword argument name.

    Args:
        model (gr.model): Model to modify
        var (dict OR gr.Marginal): Marginal information

    Returns:
        gr.model: Model with new marginals

    Examples:

        >>> import grama as gr
        >>> print(gr.valid_dist.keys()) # Supported distributions
        >>> md = gr.Model() >> \
        >>>     cp_function(
        >>>         lambda x: x[0] + x[1],
        >>>         var=["x0", "x1"],
        >>>         out=1
        >>>     ) >> \
        >>>     cp_marginals(
        >>>         x0={"dist": "norm", "loc": 0, "scale": 1}
        >>>     )

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

            new_model.density.marginals[key] = gr.MarginalNamed(
                sign=sign, d_name=dist, d_param=value_copy
            )

        ## Handle Marginal input
        if isinstance(value_copy, gr.Marginal):
            new_model.density.marginals[key] = value_copy

    new_model.update()
    return new_model


@pipe
def cp_marginals(*args, **kwargs):
    return comp_marginals(*args, **kwargs)


# Add copula
##################################################
@curry
def comp_copula_independence(model):
    r"""Add an independence copula to model

    Composition. Add an independence copula to an existing model.

    NOTE: Independence of random variables is a *very* strong assumption!
    Recommend using comp_copula_gaussian instead.

    Args:
        model (gr.model): Model to modify

    Returns:
        gr.model: Model with independence copula

        >>> import grama as gr
        >>> md = gr.Model() >> \
        >>>     cp_marginals(
        >>>         x0={"dist": "norm", "loc": 0, "scale": 1}
        >>>     ) >> \
        >>>     cp_copula_independence()

    """
    new_model = model.copy()
    new_model.density = gr.Density(
        marginals=model.density.marginals,
        copula=gr.CopulaIndependence(new_model.var_rand),
    )
    new_model.update()

    return new_model


@pipe
def cp_copula_independence(*args, **kwargs):
    return comp_copula_independence(*args, **kwargs)


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

    Examples:

        >>> import grama as gr
        >>> ## Manual assignment
        >>> md = gr.Model() >> \
        >>>     cp_marginals(
        >>>         x0={"dist": "norm", "loc": 0, "scale": 1}
        >>>         x1={"dist": "uniform", "loc": -1, "scale": 2}
        >>>     ) >> \
        >>>     cp_copula_gaussian(
        >>>         df_corr=pd.DataFrame(dict(
        >>>             var1=["x0"],
        >>>             var2=["x1"],
        >>>             corr=[0.5]
        >>>         ))
        >>>     )
        >>> ## Automated fitting
        >>> from grama.data import df_stang
        >>> md = gr.Model() >> \
        >>>     gr.cp_marginals(
        >>>         E=gr.marg_named(df_stang.E, "norm"),
        >>>         mu=gr.marg_named(df_stang.mu, "beta"),
        >>>         thick=gr.marg_named(df_stang.thick, "norm")
        >>>     ) >> \
        >>>     gr.cp_copula_gaussian(df_data=df_stang)

    """
    if not (df_corr is None):
        new_model = model.copy()
        new_model.density = gr.Density(
            marginals=model.density.marginals, copula=gr.CopulaGaussian(df_corr)
        )
        new_model.update()

        return new_model

    elif not (df_data is None):
        new_model = model.copy()
        df_corr = gr.tran_copula_corr(df_data, model=new_model)

        new_model.density = gr.Density(
            marginals=model.density.marginals, copula=gr.CopulaGaussian(df_corr)
        )
        new_model.update()

        return new_model

    else:
        raise ValueError("Must provide df_corr or df_data")


@pipe
def cp_copula_gaussian(*args, **kwargs):
    return comp_copula_gaussian(*args, **kwargs)
