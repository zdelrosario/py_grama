__all__ = [
    "eval_df",
    "ev_df",
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
from numpy import ones, eye, tile, atleast_2d
from numpy.random import seed as set_seed
from pandas import DataFrame, concat
from toolz import curry
from warnings import formatwarning, catch_warnings, simplefilter

formatwarning = custom_formatwarning

def invariants_eval_model(model):
    r"""Helper function to group common model argument invariant checks for eval functions.

    Throws errors for invalid Model inputs.

    Args:
        model (gr.Model): Model to check

    """
    ## Type Checking  
    if not isinstance(model, Model):
        if model is None:
            raise TypeError("No input model given")
        elif isinstance(model, tuple):
            raise TypeError("Given model argument is type tuple. Have you " + \
            "declared your model with an extra comma after the closing `)`?")
        else:
            raise TypeError("Type gr.Model was expected, a " + str(type(model)) + \
            " was passed.")

    ## Value checking
    if len(model.functions) == 0:
        raise ValueError("Given model has no functions")
    return   

# def valid_string_inputs(valid_strings):
#     r"""Function to turn valid string input into useful string for error
#     message. For example:
#         valid_string_inputs(["nom"]) -> 'nom'
#         valid_string_inputs(["nom", "det"]) -> 'nom' or 'det'

#     Alternatively, this could be a function to write the last half of the
#     error messages. For example:
#         func(strings_accepted = True, valid_strings = ["nom"], arg_name="df_det") ->
#             "df_det must be DataFrame or 'nom'"
#         func(strings_accepted = False, valid_strings = None, arg_name="df") ->
#             "df must be DataFrame."
#     """
#     return NotImplementedError

def invariants_eval_df(df, arg_name = "df_arg [UPDATE]", model, valid_strings = None):
    r"""Takes model input and df input as either df or [list of dfs]
    
    # Could also add an option to ignore certain tests with a lis input
    [[list of exclusions for #1][list of exclusions for #2]]
    Args:
        df (DataFrame): DataFrame to test
        model (gr.Model): SHOULD FILTER THIS TO JUST BE WHAT IS NEEDED EVENT
        valid_strings (None, list(str)): Valid string inputs 
            (such as "nom") to ignore when type testing
        arg_name (str): Name of df argument ## TO DO
    
    Examples:
        invariants_eval_df(df_det, model, ["nom", "det"])

    """
    def valid_inputs_msg(df_arg, strings_accepted, valid_strings):
        r"""Generates string explaining valid inputs for use in DataFrame
        TypeErrors and ValueErrors

        Args:
            df_arg (str): Name of df argument
            strings_accepted (bool): Indicates whether strings are accepted or not
            valid_strings (None, list(str)): Valid string inputs
        
        Returns:
            String"""
        msg = df_arg + " must be DataFrame" # general msg for valid args
        if strings_accepted: 
            # add on string options to msg
            if len(valid_strings) == 1:
                string_args = " or " + valid_strings[0]
            else:
                string_args = ", "  # comma after "must be DataFrame"
                for arg in valid_strings:
                    if arg == valid_strings[-1]:
                        # last value -> add or
                        print("last value")
                        string_args += "or '" + arg + "'"
                    else:
                        # not last value -> add comma
                        string_args += "'" + arg + "', "  # add comma                        
            msg += string_args + "."
        else: 
            # no valid string inputs, end message
            msg += "."
        return msg 

    ## Type Checking & String Input
    strings_accepted = isinstance(valid_strings, list)
    if isinstance(df, DataFrame):
        pass # input valid
    elif df is None:
        if strings_accepted:
            raise TypeError("No input df given." + arg_name +
                " must be DataFrame or " + valid_strings + ".")
        else:
            raise TypeError("No input df given" + arg_name +
                " must be DataFrame.")
    elif isinstance(df, str):
        if strings_accepted:
            if df not in valid_strings:
                raise ValueError(df_arg + " shortcut string invalid." +
                    arg_name + " must be DataFrame or " + valid_strings + ".")
        else:
            raise TypeError("Type DataFrame was expected" + 
                "<class 'str'> was passed.")
    else:
        if strings_accepted:
            raise TypeError("Invalid df input type, a " + str(type(df)) +
                " was passed."+ arg_name + " must be DataFrame or " + valid_strings + ".")
        else:
            raise TypeError("Type DataFrame was expected, a " + str(type(df)) +
                " was passed.")
    
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
    invariants_eval_model(model)
    invariants_eval_df(df, model)
    
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

## Nominal evaluation
# --------------------------------------------------
@curry
def eval_nominal(model, df_det=None, append=True, skip=False):
    r"""Evaluate model at nominal values

    Evaluates a given model at a model nominal conditions (median) of random inputs. Optionally set nominal values for the deterministic inputs.

    Args:
        model (gr.Model): Model to evaluate
        df_det (DataFrame): Deterministic levels for evaluation; use "nom" for nominal deterministic levels.
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
    ## Draw from underlying gaussian
    quantiles = ones((1, model.n_var_rand)) * 0.5  # Median

    ## Convert samples to desired marginals
    df_pr = DataFrame(data=quantiles, columns=model.var_rand)
    df_rand = model.density.pr2sample(df_pr)
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
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

    return concat(results).reset_index(drop=True)


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
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
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
    return eval_df(model, df=df_samp, append=append)


ev_conservative = add_pipe(eval_conservative)

## Random sampling
# --------------------------------------------------
@curry
def eval_sample(model, n=None, df_det=None, seed=None, append=True, skip=False, index=None):
    r"""Draw a random sample

    Evaluates a model with a random sample of the random model inputs. Generates outer product with deterministic samples.

    For more expensive models, it can be helpful to tune n to achieve a reasonable runtime. An even more effective approach is to use skip evaluation along with tran_sp() to evaluate a small, representative sample. (See examples below.)

    Args:
        model (gr.Model): Model to evaluate
        n (numeric): number of observations to draw
        df_det (DataFrame): Deterministic levels for evaluation; use "nom"
            for nominal deterministic levels.
        seed (int): random seed to use
        append (bool): Append results to input values?
        skip (bool): Skip evaluation of the functions?
        index (str or None): Name of draw index column; not added if None

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

        ## Use the draw index to facilitate plotting
        # Try running this without the `group` aesthetic in `geom_line()`;
        # without the group the plot will not have multiple lines.
        (
            md_beam
            >> gr.ev_sample(
                n=20,
                df_det=gr.df_make(w=3, t=gr.linspace(2, 4, 100)),
                index="idx",
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
    if n is None:
        raise ValueError("Must provide a valid n value.")

    ## Set seed only if given
    if seed is not None:
        set_seed(seed)

    ## Ensure sample count is int
    if not isinstance(n, Integral):
        print("eval_sample() is rounding n...")
        n = int(n)

    ## Draw samples
    df_rand = model.density.sample(n=n, seed=seed)
    if not index is None:
        df_rand[index] = df_rand.index
    ## Construct outer-product DOE
    df_samp = model.var_outer(df_rand, df_det=df_det)

    if skip:
        ## Evaluation estimate
        runtime_est = model.runtime(df_samp.shape[0])
        if runtime_est > 0:
            print(
                "Estimated runtime for design with model ({0:1}):\n  {1:4.3} sec".format(
                    model.name, runtime_est
                )
            )
        else:
            print("Design runtime estimates unavailable; model has no timing data.")

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
