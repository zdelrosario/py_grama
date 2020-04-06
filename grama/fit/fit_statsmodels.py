__all__ = ["fit_ols", "ft_ols"]

## Fitting via statsmodels package
from numpy import zeros
from pandas import DataFrame

try:
    import statsmodels.formula.api as smf

except ModuleNotFoundError:
    raise ModuleNotFoundError("module statsmodels not found")

import grama as gr
from grama import pipe
from toolz import curry

## Fit model via OLS
# --------------------------------------------------
@curry
def fit_ols(df, formulae=[""], domain=None, density=None):
    """Fit a function via Ordinary Least Squares

    Fit a function via ordinary least squares. Specify features via
    statsmodels formula.

    Args:
        df (DataFrame): Data for function fitting
        formulae (list(str)): List of statsmodels formulae
        domain (gr.Domain): Domain for new model
        density (gr.Density): Density for new model

    Returns:
        gr.Model: A grama model with fitted function(s)

    @pre domain is not None
    @pre len(formulae) == len(domain.inputs)

    Notes:
        - Wrapper for statsmodels.formula.api.ols

    """
    n_obs, n_in = df.shape

    ## Parse formulae for output names
    n_out = len(formulae)
    outputs = [""] * n_out
    for ind in range(n_out):
        ind_start = formulae[ind].find("~")
        outputs[ind] = formulae[ind][:ind_start].strip()

    ## Construct fits
    fits = []
    for ind in range(n_out):
        fits.append(smf.ols(formulae[ind], data=df).fit())

    def fit_all(df_new):
        n_obs_new, _ = df_new.shape
        result = zeros((n_obs_new, n_out))
        for ind in range(n_out):
            result[:, ind] = fits[ind].predict(df_new)
        return DataFrame(data=result, columns=outputs)

    ## Construct model
    return gr.model_vectorized(
        function=fit_all, outputs=outputs, domain=domain, density=density
    )


@pipe
def ft_ols(*args, **kwargs):
    return fit_ols(*args, **kwargs)
