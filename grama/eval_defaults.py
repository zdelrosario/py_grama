__all__ = [
    "eval_df",
    "ev_df",
    "eval_linup",
    "ev_linup",
    "eval_nominal",
    "ev_nominal",
    "eval_grad_fd",
    "ev_grad_fd",
    "eval_sample",
    "ev_sample",
    "eval_conservative",
    "ev_conservative",
]

import itertools
from grama import add_pipe, tran_outer, custom_formatwarning, Model
from numbers import Integral
from numpy import diag, dot, ones, eye, tile, atleast_2d, triu_indices, round
from numpy.random import seed as set_seed
from pandas import DataFrame, concat
from toolz import curry
from warnings import formatwarning, catch_warnings, simplefilter

formatwarning = custom_formatwarning

def invariants_eval_model(md, skip=False):
    r"""Helper function to group common model argument invariant checks for eval functions.

    Throws errors for invalid Model inputs.

    Args:
        md (gr.Model): Model to check
        skip (bool): if function is skipping evaluation of function. If True,
            skips model.functions test

    """
    ## Type Checking
    if not isinstance(md, Model):
        if md is None:
            raise TypeError("No input model given")
        elif isinstance(md, tuple):
            raise TypeError("Given model argument is type tuple. Have you " +
            "declared your model with an extra comma after the closing `)`?")
        else:
            raise TypeError("Type gr.Model was expected, a " + str(type(md)) +
            " was passed.")

    ## Value checking
    if not skip and len(md.functions) == 0:
        raise ValueError("Given model has no functions.")
    return

def invariants_eval_df(df, arg_name="df", valid_str=None, acc_none=False):
    r"""Helper function to group common DataFrame argument invariant checks for eval functions.

    Throws errors for invalid DataFrame inputs.

    Args:
        df (DataFrame): DataFrame to test
        arg_name (str): Name of df argument
        valid_str (list(str) or None): Valid string inputs, if any, to
            allow when testing
        acc_none (bool): allow `None` as a valid df input

    Examples:
        invariants_eval_df(df)
        invariants_eval_df(df_det, arg_name="df_det", valid_str=["nom", "det"])
        invariants_eval_df(df_test, arg_name="df_test", acc_none=())

    """
    def aps(string):
        r"""Helper function for valid_args_msg() to put apostrophes around a
        string.

        Args:
            string (str): string to surround

        Returns:
            String: Input string surrounded as 'input'"""
        return "'" + string + "'"

    def valid_args_msg(df_arg, acc_str, valid_str):
        r"""Generates string explaining valid inputs for use in DataFrame
        TypeErrors and ValueErrors

        Args:
            df_arg (str): Name of df argument
            acc_str (bool): Indicates whether strings are accepted or not
            valid_str (None, list(str)): Valid string inputs

        Returns:
            String"""
        msg = df_arg + " must be DataFrame" # general msg for valid args
        if acc_str:
            # add on string options to msg
            if len(valid_str) == 1:
                string_args = " or " + aps(valid_str[0])
            else:
                string_args = ", "  # comma after "must be DataFrame"
                for arg in valid_str:
                    if arg == valid_str[-1]:
                        # last value -> add or
                        string_args += "or " + aps(arg)
                    else:
                        # not last value -> add comma
                        string_args += aps(arg) + ", "  # add comma
            msg += string_args + "."
        else:
            # no valid string inputs, end message
            msg += "."
        return msg

    ## Type Checking & String Input
    acc_str = isinstance(valid_str, list)
    if not isinstance(df, DataFrame):
        if df is None:
            if not acc_none:
                # allow "None" df if None accepted
                raise TypeError("No " + arg_name + " argument given. " +
                    valid_args_msg(arg_name, acc_str, valid_str))
        elif isinstance(df, str) and acc_str:
            # case check for invalid str input
            if df not in valid_str:
                raise ValueError(arg_name + " shortcut string invalid. " +
                    valid_args_msg(arg_name, acc_str, valid_str))
        else:
            raise TypeError(valid_args_msg(arg_name, acc_str, valid_str) +
                " Given argument is type " + str(type(df)) +
                    ". ")

    ## Value checking
    #### TO DO

    return



## Default evaluation function
# --------------------------------------------------
@curry
def eval_df(model, df=None, append=True, verbose=True):
    r"""Evaluate model at given values

    Evaluates a given model at a given dataframe.

    Args:
        model (gr.Model): Model to evaluate
        df (DataFrame): Input dataframe to evaluate
        append (bool): Append results to original dataframe?

    Returns:
        DataFrame: Results of model evaluation

    Examples::

        import grama as gr
        from grama.models import make_test
        md = make_test()
        df = gr.df_make(x0=0, x1=1, x2=2)
        md >> gr.ev_df(df=df)

    """
    ## Perform common invariant tests
    invariants_eval_model(model)
    invariants_eval_df(df)

    out_intersect = set(df.columns).intersection(model.out)
    if (len(out_intersect) > 0) and verbose:
        print(
            "... provided columns intersect model output.\n"
            + "eval_df() is dropping {}".format(out_intersect)
        )

    df_res = model.evaluate_df(df)

    if append:
        df_res = concat(
            [
                df.reset_index(drop=True).drop(model.out, axis=1, errors="ignore"),
                df_res,
            ],
            axis=1,
        )

    return df_res


ev_df = add_pipe(eval_df)

## Linear Uncertainty Propagation
# --------------------------------------------------
@curry
def eval_linup(model, df_base=None, append=True, decomp=False, decimals=2, n=1e4, seed=None):
    r"""Linear uncertainty propagation

    Approximates the variance of output models using a linearization of functions---linear uncertainty propagation. Optionally decomposes the output variance according to additive factors from each input.

    Args:
        model (gr.Model): Model to evaluate
        df_base (DataFrame or None): Base levels for evaluation; use
            "nom" for nominal levels.
        append (bool): Append results to nominal inputs?
        decomp (bool): Decompose the fractional variances according to each input?
        decimals (int): Decimals to report for fractional variances

        n (float): Monte Carlo sample size, for estimating covariance matrix
        seed (int or None): Monte Carlo seed

    Returns:
        DataFrame: Output variances at each deterministic level

    Examples::

        import grama as gr
        from grama.models import make_test
        md = make_test()
        ## Set manual levels for deterministic inputs; nominal levels for random inputs
        md >> gr.ev_linup(df_det=gr.df_make(x2=[0, 1, 2])
        ## Use nominal deterministic levels
        md >> gr.ev_linup(df_base="nom")

    """
    ## Perform common invariant tests
    invariants_eval_model(model, False)
    invariants_eval_df(df_base, arg_name="df_base", valid_str=["nom"])

    if df_base is "nom":
        df_base = eval_nominal(model, df_det="nom")
    else:
        df_base = eval_df(model, df=df_base)

    ## Approximate the covariance matrix
    df_sample = eval_sample(model, n=n, seed=seed, df_det=df_base[model.var_det], skip=True)
    cov = df_sample[model.var_rand].cov()

    ## Approximate the gradient
    df_grad = eval_grad_fd(model, df_base=df_base, var="rand", append=True)

    ## Iterate over all outputs and gradient points
    df_res = DataFrame({})
    for out in model.out:
        for i in range(df_grad.shape[0]):
            # Build gradient column names
            var_names = map(
                lambda s: "D" + out + "_D" + s,
                model.var_rand
            )
            # Extract gradient values
            grad = df_grad.iloc[i][var_names].values.flatten()
            # Approximate variance
            var = dot(grad, dot(cov, grad))

            # Store values
            df_tmp = df_base.iloc[[i]][model.var + model.out].reset_index(drop=True)
            df_tmp["out"] = out
            df_tmp["var"] = var

            # Decompose variance due to each input
            if decomp:
                grad_mat = diag(grad)
                sens_mat = dot(grad_mat, dot(cov, grad_mat)) / var
                U, V = triu_indices(sens_mat.shape[0])
                sens_values = [
                    round(
                        sens_mat[U[j], V[j]] * (1 + (U[j] != V[j])), # Double off-diag
                        decimals=decimals
                    )
                    for j in range(len(U))
                ]
                sens_var = [
                    model.var_rand[U[j]] + "*" + model.var_rand[V[j]]
                    for j in range(len(U))
                ]

                df_sens = DataFrame({
                    "var_frac": sens_values,
                    "var_rand": sens_var
                })
                df_tmp = tran_outer(df_tmp, df_sens)

            df_res = concat(
                (df_res,df_tmp),
                axis=0,
            ).reset_index(drop=True)

    return df_res

ev_linup = add_pipe(eval_linup)


## Nominal evaluation
# --------------------------------------------------
@curry
def eval_nominal(model, df_det=None, append=True, skip=False):
    r"""Evaluate model at nominal values

    Evaluates a given model at a model nominal conditions (median) of random inputs. Optionally set nominal values for the deterministic inputs.

    Args:
        model (gr.Model): Model to evaluate
        df_det (DataFrame or None): Deterministic levels for evaluation; use
            "nom" for nominal deterministic levels. If provided model has no
            deterministic variables (model.n_var_det == 0), then df_det may
            equal None.
        append (bool): Append results to nominal inputs?
        skip (bool): Skip evaluation of the functions?

    Returns:
        DataFrame: Results of nominal model evaluation or unevaluated design

    Examples::

        import grama as gr
        from grama.models import make_test
        md = make_test()
        ## Set manual levels for deterministic inputs; nominal levels for random inputs
        md >> gr.ev_nominal(df_det=gr.df_make(x2=[0, 1, 2])
        ## Use nominal deterministic levels
        md >> gr.ev_nominal(df_det="nom")

    """
    ## Perform common invariant tests
    invariants_eval_model(model, skip)
    invariants_eval_df(df_det, arg_name="df_det", valid_str=["nom"],
        acc_none=(model.n_var_det==0))

    ## Draw from underlying gaussian
    quantiles = ones((1, model.n_var_rand)) * 0.5  # Median

    ## Convert samples to desired marginals
    df_pr = DataFrame(data=quantiles, columns=model.var_rand)
    df_rand = model.density.pr2sample(df_pr)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        ## Evaluation estimate
        runtime_msg = model.runtime_message(df_samp)
        print(runtime_msg)

        return df_samp
    return eval_df(model, df=df_samp, append=append)


ev_nominal = add_pipe(eval_nominal)

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

    Examples::

        import grama as gr
        from grama.models import make_cantilever_beam
        md = make_cantilever_beam()
        # Select base point(s)
        df_nom = md >> gr.ev_nominal(df_det="nom")
        # Approximate the gradient
        df_grad = md >> gr.ev_grad_fd(df_base=df_nom)

    """
    ## Check invariants
    invariants_eval_model(model, skip)
    invariants_eval_df(df_base, arg_name="df_base")
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
            tran_outer(
                DataFrame(
                    columns=var, data=-stencil + df_base[var].iloc[[row_i]].values
                ),
                df_base[var_fix].iloc[[row_i]],
            ),
            append=False,
        )

        df_right = eval_df(
            model,
            tran_outer(
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

    # Consolidate results
    df_res = concat(results).reset_index(drop=True)

    if append:
        return concat((df_base, df_res), axis=1)
    else:
        return df_res


ev_grad_fd = add_pipe(eval_grad_fd)

## Conservative quantile evaluation
# --------------------------------------------------
@curry
def eval_conservative(model, quantiles=None, df_det=None, append=True, skip=False):
    r"""Evaluates a given model at conservative input quantiles

    Uses model specifications to determine the "conservative" direction for each input, and evaluates the model at the desired quantile. Provided primarily for comparing UQ against pseudo-deterministic design criteria (del Rosario et al.; 2021).

    Note that if there is no conservative direction for the given input, the given quantile will be ignored and the median will automatically be selected.

    Args:
        model (gr.Model): Model to evaluate
        quantiles (numeric): lower quantile value(s) for conservative
            evaluation; can be single value for all inputs, array
            of values for each random variable, or None for default 0.01.
            values in [0, 0.5]
        df_det (DataFrame or None): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels. If provided model has no
            deterministic variables (model.n_var_det == 0), then df_det may
            equal None.
        append (bool): Append results to conservative inputs?
        skip (bool): Skip evaluation of the functions?

    Returns:
        DataFrame: Conservative evaluation or unevaluated design

    References:
        del Rosario, Zachary, Richard W. Fenrich, and Gianluca Iaccarino. "When Are Allowables Conservative?." AIAA Journal 59.5 (2021): 1760-1772.

    Examples::

        import grama as gr
        from grama.models import make_plate_buckle
        md = make_plate_buckle()
        # Evaluate at conservative input values
        md >> gr.ev_conservative(df_det="nom")

    """
    ## Check invariants
    invariants_eval_model(model, skip)
    invariants_eval_df(df_det, arg_name="df_det", valid_str=["nom"],
        acc_none=(model.n_var_det==0))

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
        ## Evaluation estimate
        runtime_msg = model.runtime_message(df_samp)
        print(runtime_msg)

        return df_samp
    return eval_df(model, df=df_samp, append=append)


ev_conservative = add_pipe(eval_conservative)

## Random sampling
# --------------------------------------------------
@curry
def eval_sample(model, n=None, df_det=None, seed=None, append=True, skip=False, comm=True, ind_comm=None):
    r"""Draw a random sample

    Evaluates a model with a random sample of the random model inputs. Generates outer product with deterministic levels (common random numbers) OR generates a sample fully-independent of deterministic levels (non-common random numbers).

    For more expensive models, it can be helpful to tune n to achieve a reasonable runtime. An even more effective approach is to use skip evaluation along with tran_sp() to evaluate a small, representative sample. (See examples below.)

    Args:
        model (gr.Model): Model to evaluate
        n (numeric): number of observations to draw
        df_det (DataFrame or None): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels. If provided model has no
            deterministic variables (model.n_var_det == 0), then df_det may
            equal None.
        seed (int): random seed to use
        append (bool): Append results to input values?
        skip (bool): Skip evaluation of the functions?
        comm (bool): Use common random numbers (CRN) across deterministic levels? CRN will tend to aid in the comparison of statistics across deterministic levels and enhance the convergence of stochastic optimization.
        ind_comm (str or None): Name of realization index column; not added if None

    Returns:
        DataFrame: Results of evaluation or unevaluated design

    Examples::

        import grama as gr
        from grama.models import make_test
        DF = gr.Intention()

        # Simple random sample evaluation
        md = make_test()
        df = md >> gr.ev_sample(n=1e2, df_det="nom")
        df.describe()

        ## Use autoplot to visualize results
        (
            md
            >> gr.ev_sample(n=1e2, df_det="nom")
            >> gr.pt_auto()
        )

        ## Cantilever beam examples
        from grama.models import make_cantilever_beam
        md_beam = make_cantilever_beam()

        ## Use the realization index to facilitate plotting
        # Try running this without the `group` aesthetic in `geom_line()`;
        # without the group the plot will not have multiple lines.
        (
            md_beam
            >> gr.ev_sample(
                n=20,
                df_det=gr.df_make(w=3, t=gr.linspace(2, 4, 100)),
                ind_comm="idx",
            )

            >> gr.ggplot(gr.aes("t", "g_stress"))
            + gr.geom_line(gr.aes(color="w", group="idx"))
        )

        ## Use iocorr to generate input/output correlation tile plot
        (
            md_beam
            >> gr.ev_sample(n=1e3, df_det="nom", skip=True)
            # Generate input/output correlation summary
            >> gr.tf_iocorr()
            # Visualize
            >> gr.pt_auto()
        )

        ## Use support points to reduce model runtime
        (
            md_beam
            # Generate large input sample but don't evaluate outputs
            >> gr.ev_sample(n=1e5, df_det="nom", skip=True)
            # Reduce to a smaller---but representative---sample
            >> gr.tf_sp(n=50)
            # Evaluate the outputs
            >> gr.tf_md(md_beam)
        )

        ## Estimate probabilities
        (
            md_beam
            # Generate large
            >> gr.ev_sample(n=1e5, df_det="nom")
            # Estimate probabilities of failure
            >> gr.tf_summarize(
                pof_stress=gr.mean(DF.g_stress <= 0),
                pof_disp=gr.mean(DF.g_disp <= 0),
            )
        )


    """
    ## Check invariants
    invariants_eval_model(model, skip)
    invariants_eval_df(df_det, arg_name="df_det", valid_str=["nom"],
        acc_none=(model.n_var_det==0))
    if n is None:
        raise ValueError("Must provide a valid n value.")

    ## Set seed only if given
    if seed is not None:
        set_seed(seed)

    ## Ensure sample count is int
    if not isinstance(n, Integral):
        print("eval_sample() is rounding n...")
        n = int(n)

    ## Draw realizations
    # Common random numbers
    if comm:
        df_rand = model.density.sample(n=n, seed=seed)
        if not ind_comm is None:
            df_rand[ind_comm] = df_rand.index
        df_samp = model.var_outer(df_rand, df_det=df_det)
    # Non-common random numbers
    else:
        df_rand = model.density.sample(n=n * df_det.shape[0], seed=seed)
        if not ind_comm is None:
            df_rand[ind_comm] = df_rand.index
        df_samp = concat(
            (df_rand, concat([df_det[model.var_det]]*n, axis=0).reset_index(drop=True)),
            axis=1,
        ).reset_index(drop=True)


    if skip:
        ## Evaluation estimate
        runtime_msg = model.runtime_message(df_samp)
        print(runtime_msg)

        ## Attach metadata
        with catch_warnings():
            simplefilter("ignore")
            df_samp._plot_info = {
                "type": "sample_inputs",
                "var": model.var_rand,
            }

        return df_samp


    df_res = eval_df(model, df=df_samp, append=append)
    ## Attach metadata
    with catch_warnings():
        simplefilter("ignore")
        df_res._plot_info = {
            "type": "sample_outputs",
            "var": model.var,
            "out": model.out,
        }

    return df_res


ev_sample = add_pipe(eval_sample)
