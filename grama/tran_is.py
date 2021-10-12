__all__ = [
    "tran_reweight",
    "tf_reweight",
]

from grama import add_pipe, pipe, custom_formatwarning
from toolz import curry

## Reweight using likelihood ratio
# --------------------------------------------------
@curry
def tran_reweight(df_base, md_base, md_new, var_weight="weight"):
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
        md_base (Model): Model used to generate `df`.
        md_new (Model): Model defining a new "What If?" scenario.
        var_weight (string): Name to give new column of weights.

    Returns:
        DataFrame: Original df_base with added column of weights.

    Notes:
        - The base scenario `md_base` should have fatter tails than any of
          the scenarios considered as `df_new`. See Owen (2013) Chapter 9
          for more details.

    References:
        A.B. Owen, "Monte Carlo theory, methods and examples" (2013)

    Examples:

    """
    pass
