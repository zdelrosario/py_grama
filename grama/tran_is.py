__all__ = [
    "tran_reweight",
    "tf_reweight",
]

from grama import add_pipe, pipe, custom_formatwarning
from pandas import concat, DataFrame
from toolz import curry

## Reweight using likelihood ratio
# --------------------------------------------------
@curry
def tran_reweight(
        df_base,
        md_base,
        md_new,
        var_weight="weight",
        append=True,
):
    r"""Reweight a sample using likelihood ratio

    Reweight is a tool to facilitate "What If?" Monte Carlo simulation;
    specifically, to make testing a models with the same function(s) but
    different distributions more computationally efficient.

    This tool automates calulation of the *likelihood ratio* between the
    distributions of two given models. Using the resulting weights to scale
    (elementwise multiply) output values and compute summaries is called
    *importance sampling*, enabling "What If?" testing. Use of this tool enables
    one to generate a single Monte Carlo sample, rather than multiple samples
    for each "What If?" scenario (avoiding extraneous function evaluations).

    Let `y` be a generic output of the scenario. The importance sampling
    procedure is as follows:

    1. Create a base scenario represented by `md_base`, and a desired number
       of alternative "What If?" scenarios represented my other models.
    2. Use `gr.eval_monte_carlo` to generate a single sample `df_base` of size `n`
       using the base scenario `md_base`. Compute statistics of interest on
       the output `y` for this base scenario.
    3. For each alternative scenario `md_new`, use `gr.tran_reweight` to
       generate weights `weight`, and use the tool `n_e = gr.neff_is(DF.weight)`
       to compute the effective sample size. If `n_e << n`, then importance
       sampling is unlikely to give an accurate estimate for this scenario.
    4. For each alternative scenario `md_new`, use the relevant weights
       `weight` to scale the output value `y_new = y * weight`, and compute
       statistics of interest of the weighted output values.

    Args:
        df_base (DataFrame): Monte Carlo results from `md_base`.
        md_base (Model): Model used to generate `df_base`.
        md_new (Model): Model defining a new "What If?" scenario.
        var_weight (string): Name to give new column of weights.
        append (bool): Append results to original DataFrame?

    Returns:
        DataFrame: Original df_base with added column of weights.

    Notes:
        - The base scenario `md_base` should have fatter tails than any of
          the scenarios considered as `df_new`. See Owen (2013) Chapter 9
          for more details.

    References:
        A.B. Owen, "Monte Carlo theory, methods and examples" (2013)

    Examples:

        >>> import grama as gr
        >>> from grama.models import make_cantilever_beam
        >>> DF = gr.Intention()
        >>>
        >>> md_base = make_cantilever_beam()
        >>> md_new = (
        >>>     md_base
        >>>     >> gr.cp_marginals(
        >>>         H=dict(dist="norm", loc=500.0, scale=50.0),
        >>>     )
        >>> )
        >>>
        >>> ## Assess safety via simple Monte Carlo
        >>> df_base = gr.eval_monte_carlo(md_base, df_det="nom", n=1e3)
        >>> print(
        >>>     df_base
        >>>     >> gr.tf_summarize(
        >>>         pof_stress=gr.mean(DF.g_stress <= 0),
        >>>         pof_disp=gr.mean(DF.g_disp <= 0),
        >>>     )
        >>> )
        >>>
        >>> ## Re-use samples to test another scenario
        >>> print(
        >>>     df_base
        >>>     >> gr.tf_reweight(md_base=md_base, md_new=md_new)
        >>>     >> gr.tf_summarize(
        >>>         pof_stress=gr.mean((DF.g_stress <= 0) * DF.weight),
        >>>         pof_disp=gr.mean((DF.g_disp <= 0) * DF.weight),
        >>>         n_eff=gr.neff_is(DF.weight),
        >>>     )
        >>> )
        >>>
        >>> ## It is unsafe to study new scenarios with wider uncertainty than the base
        >>> ## scenario
        >>> md_poor = (
        >>>     md_base
        >>>     >> gr.cp_marginals(
        >>>         H=dict(dist="norm", loc=500.0, scale=400.0),
        >>>     )
        >>> )
        >>> ## Note the tiny effective size in this case
        >>> print(
        >>>     md_base
        >>>     >> gr.ev_monte_carlo(n=1e3, df_det="nom")
        >>>     >> gr.tf_reweight(md_base=md_base, md_new=md_poor)
        >>>     >> gr.tf_summarize(
        >>>         pof_stress=gr.mean((DF.g_stress <= 0) * DF.weight),
        >>>         pof_disp=gr.mean((DF.g_disp <= 0) * DF.weight),
        >>>         n_eff=gr.neff_is(DF.weight),
        >>>     )
        >>> )

    """
    ## Check invariants
    # Check that random inputs to md_base available in df_base
    var_diff = set(md_base.var_rand).difference(set(df_base.columns))
    if not (len(var_diff) == 0):
        raise ValueError(
            "Random variables in md_base missing from df_base:\n" +
            "Missing: {}".format(var_diff)
        )
    # Check that random inputs match between models
    var_base = set(md_base.var_rand)
    var_new = set(md_new.var_rand)
    if var_base != var_new:
        raise ValueError(
            "Random variables of md_base and md_var must match:\n" +
            "md_base is missing: {}\n".format(var_new.difference(var_base)) +
            "md_new is missing: {}".format(var_base.difference(var_new))
        )
    # Check that deterministic inputs match between models
    var_base = set(md_base.var_det)
    var_new = set(md_new.var_det)
    if var_base != var_new:
        raise ValueError(
            "Deterministic variables of md_base and md_var must match:\n" +
            "md_base is missing: {}\n".format(var_new.difference(var_base)) +
            "md_new is missing: {}".format(var_base.difference(var_new))
        )
    # Check that `weights` name does not collide
    if (var_weight in df_base.columns) and append:
        raise ValueError(
            "Weight name {} already in df_base.columns; ".format(var_weight) +
            "choose a new name."
        )

    ## Compute weight values
    # Use base model for importance distribution
    q = md_base.density.d(df_base)
    # Use new model for nominal distribution
    p = md_new.density.d(df_base)
    # Compute likelihood ratio
    w = p / q

    ## Return results
    df_res = DataFrame({var_weight : w})

    if append:
        df_res = concat(
            [df_base.reset_index(drop=True), df_res],
            axis=1,
        )

    return df_res

tf_reweight = add_pipe(tran_reweight)
