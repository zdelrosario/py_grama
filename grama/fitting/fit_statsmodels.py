## Fitting via statsmodels package
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from .. import core
from toolz import curry

## Fit model via OLS
@curry # TODO Determine why pipe fails with this function....
def ft_ols(df, formulae = [""], domain = None, density = None):
    """Fit a model via Ordinary Least Squares

    @param df pandas dataframe
    @param formulae set of R-like formula strings
    @param domain domain object for new model
    @param density density object for new model

    @return model a valid grama model

    @pre domain is not None
    @pre len(formulae) == len(domain.inputs)
    """

    ## Check invariants
    if domain is None:
        raise ValueError("No domain given")

    n_obs, n_in = df.shape

    ## Parse formulae for output names
    n_out   = len(formulae)
    outputs = [""] * n_out
    for ind in range(n_out):
        ind_start = formulae[ind].find("~")
        outputs[ind] = formulae[ind][:ind_start].strip()

    ## Construct fits
    fits = []
    for ind in range(n_out):
        fits.append(smf.ols(formulae[ind], data = df).fit())

    def fit_all(df_new):
        n_obs_new, _ = df_new.shape
        result = np.zeros((n_obs_new, n_out))
        for ind in range(n_out):
            result[:, ind] = fits[ind].predict(df_new)
        return pd.DataFrame(data = result, columns = outputs)

    ## Construct model
    return core.model_df_(
        function = fit_all,
        outputs  = outputs,
        domain   = domain,
        density  = density
    )
