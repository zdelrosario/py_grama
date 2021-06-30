__all__ = [
    "fit_lolo",
    "ft_lolo",
]

## Fitting via sklearn package
try:
    from lolopy.learners import RandomForestRegressor

except ModuleNotFoundError:
    raise ModuleNotFoundError("module lolopy not found")

import grama as gr
from grama import add_pipe, pipe
from numpy import stack
from numpy.random import seed as set_seed
from pandas import DataFrame
from toolz import curry
from warnings import filterwarnings


## Helper functions and classes
# --------------------------------------------------
class FunctionRFR(gr.Function):
    def __init__(self, rf, var, out, name, runtime, return_std):
        """

        Args:
            rf (scikit RandomForestRegressor):
        """
        self.rf = rf
        self.var = var
        self.name = name
        self.runtime = runtime

        self.return_std = return_std
        if return_std:
            self.out = list(map(lambda s: s + "_mean", out)) + list(map(lambda s: s + "_sd", out))
        else:
            self.out = list(map(lambda s: s + "_mean", out))

    def eval(self, df):
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.var).issubset(set(df.columns)):
            raise ValueError(
                "Model function `{}` var not a subset of given columns".format(
                    self.name
                )
            )

        ## Predict
        if self.return_std:
            res = self.rf.predict(df[self.var], return_std=True)
            y = stack((res[0], res[1])).T
        else:
            y = self.rf.predict(df[self.var])

        return DataFrame(data=y, columns=self.out)


## Fit random forest model with sklearn
# --------------------------------------------------
@curry
def fit_lolo(
    df,
    md=None,
    var=None,
    out=None,
    domain=None,
    density=None,
    seed=None,
    return_std=True,
    suppress_warnings=True,
    **kwargs
):
    r"""Fit a random forest

    Fit a random forest to given data. Specify inputs and outputs, or inherit
    from an existing model.

    Args:
        df (DataFrame): Data for function fitting
        md (gr.Model): Model from which to inherit metadata
        var (list(str) or None): List of features or None for all except outputs
        out (list(str)): List of outputs to fit
        domain (gr.Domain): Domain for new model
        density (gr.Density): Density for new model
        seed (int or None): Random seed for fitting process
        return_std (bool): Return predictive standard deviations?
        suppress_warnings (bool): Suppress warnings when fitting?

    Keyword Arguments:

        num_trees (int):
        use_jackknife (bool):
        bias_learner ():
        leaf_learner ():
        subset_strategy (str):
        min_leaf_instances (int):
        max_depth (int):
        uncertainty_calibration (bool):
        randomize_pivot_location (bool):
        randomly_rotate_features (bool):

    Returns:
        gr.Model: A grama model with fitted function(s)

    Notes:
        - Wrapper for lolopy.learners.RandomForestRegressor

    """
    if suppress_warnings:
        filterwarnings("ignore")

    n_obs, n_in = df.shape

    ## Check minimum rows
    if n_obs < 8:
        raise ValueError("The lolo random forest requires at least 8 rows")

    ## Infer fitting metadata, if available
    if not (md is None):
        domain = md.domain
        density = md.density
        out = md.out

    ## Check invariants
    if not set(out).issubset(set(df.columns)):
        raise ValueError("out must be subset of df.columns")
    ## Default input value
    if var is None:
        var = list(set(df.columns).difference(set(out)))
    ## Check more invariants
    set_inter = set(out).intersection(set(var))
    if len(set_inter) > 0:
        raise ValueError(
            "outputs and inputs must be disjoint; intersect = {}".format(set_inter)
        )
    if not set(var).issubset(set(df.columns)):
        raise ValueError("var must be subset of df.columns")

    ## Construct gaussian process for each output
    functions = []

    for output in out:
        rf = RandomForestRegressor(**kwargs)
        set_seed(seed)
        rf.fit(df[var].values, df[output].values)
        name = "RF"

        fun = FunctionRFR(rf, var, [output], name, 0, return_std)
        functions.append(fun)

    ## Construct model
    return gr.Model(functions=functions, domain=domain, density=density)


ft_lolo = add_pipe(fit_lolo)
