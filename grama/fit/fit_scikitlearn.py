__all__ = [
    "fit_gp",
    "ft_gp",
    "fit_lm",
    "ft_lm",
    "fit_rf",
    "ft_rf",
    "fit_kmeans",
    "ft_kmeans",
]

## Fitting via sklearn package
try:
    from sklearn.base import clone
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Kernel, RBF, ConstantKernel as Con
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor

except ModuleNotFoundError:
    raise ModuleNotFoundError("module sklearn not found")

import grama as gr
from copy import deepcopy
from grama import add_pipe, pipe
from pandas import concat, DataFrame, Series
from toolz import curry
from warnings import filterwarnings


## Helper functions and classes
# --------------------------------------------------
def standardize_cols(df, ser_min, ser_max, var):
    """
    @pre set(ser_min.index) == set(ser_max.index)
    """

    df_std = df.copy()
    for v in var:
        den = ser_max[v] - ser_min[v]
        if den < 1e-16:
            den = 1
        df_std[v] = (df_std[v] - ser_min[v]) / den
    return df_std


def restore_cols(df, ser_min, ser_max, var):
    """
    @pre set(ser_min.index) == set(ser_max.index)
    """
    df_res = df.copy()
    for v in var:
        den = ser_max[v] - ser_min[v]
        if den < 1e-16:
            den = 1
        df_res[v] = den * df[v] + ser_min[v]
    return df_res


class FunctionGPR(gr.Function):
    def __init__(self, gpr, var, out, name, runtime, var_min, var_max):
        self.gpr = gpr
        # self.df_train = df_train
        self.var = var
        ## "Natural" outputs; what we're modeling
        self.out_nat = out
        ## Predicted outputs; mean and std
        self.out_mean = list(map(lambda s: s + "_mean", out))
        self.out_sd = list(map(lambda s: s + "_sd", out))
        self.out = self.out_mean + self.out_sd

        self.name = name
        self.runtime = runtime

        self.var_min = var_min
        self.var_max = var_max

    def eval(self, df):
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.var).issubset(set(df.columns)):
            raise ValueError(
                "Model function `{}` var not a subset of given columns".format(
                    self.name
                )
            )
        df_sd = standardize_cols(df, self.var_min, self.var_max, self.var)
        y, y_sd = self.gpr.predict(df_sd[self.var], return_std=True)

        return concat(
            (
                DataFrame(data=y, columns=self.out_mean),
                DataFrame(data=y_sd, columns=self.out_sd),
            ),
            axis=1,
        )

    def copy(self):
        func_new = FunctionGPR(
            self.gpr,
            self.df_train.copy(),
            self.var,
            self.out_nat,
            self.name,
            self.runtime,
        )

        return func_new


class FunctionRegressor(gr.Function):
    def __init__(self, regressor, var, out, name, runtime):
        """

        Args:
            regressor (scikit Regressor):
        """
        self.regressor = regressor
        self.var = var
        self.out = list(map(lambda s: s + "_mean", out))
        self.name = name
        self.runtime = runtime

    def eval(self, df):
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.var).issubset(set(df.columns)):
            raise ValueError(
                "Model function `{}` var not a subset of given columns".format(
                    self.name
                )
            )

        ## Predict
        y = self.regressor.predict(df[self.var])
        return DataFrame(data=y, columns=self.out)


## Fit GP model with sklearn
# --------------------------------------------------
@curry
def fit_gp(
    df,
    md=None,
    var=None,
    out=None,
    domain=None,
    density=None,
    kernels=None,
    seed=None,
    suppress_warnings=True,
    n_restart=5,
    alpha=1e-10,
):
    r"""Fit a gaussian process

    Fit a gaussian process to given data. Specify var and out, or inherit from
    an existing model.

    Note that the new model will have two outputs `y_mean, y_sd` for each
    original output `y`. The quantity `y_mean` is the best-fit value, while
    `y_sd` is a measure of predictive uncertainty.

    Args:
        df (DataFrame): Data for function fitting
        md (gr.Model): Model from which to inherit metadata
        var (list(str) or None): List of features or None for all except outputs
        out (list(str)): List of outputs to fit
        domain (gr.Domain): Domain for new model
        density (gr.Density): Density for new model
        seed (int or None): Random seed for fitting process
        kernels (sklearn.gaussian_process.kernels.Kernel or dict or None): Kernel for GP
        n_restart (int): Restarts for optimization
        alpha (float or iterable): Value added to diagonal of kernel matrix
        suppress_warnings (bool): Suppress warnings when fitting?

    Returns:
        gr.Model: A grama model with fitted function(s)

    Notes:
        - Wrapper for sklearn.gaussian_process.GaussianProcessRegressor

    """
    if suppress_warnings:
        filterwarnings("ignore")

    n_obs, n_in = df.shape

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
            "out and var must be disjoint; intersect = {}".format(set_inter)
        )
    if not set(var).issubset(set(df.columns)):
        raise ValueError("var must be subset of df.columns")

    ## Pre-process kernel selection
    if kernels is None:
        # Vectorize
        kernels = {o: None for o in out}
    elif isinstance(kernels, Kernel):
        kernels = {o: kernels for o in out}

    ## Pre-process data
    var_min = df[var].min()
    var_max = df[var].max()
    df_sd = standardize_cols(df, var_min, var_max, var)

    ## Construct gaussian process for each output
    functions = []

    for output in out:
        # Define and fit model
        gpr = GaussianProcessRegressor(
            kernel=deepcopy(kernels[output]),
            random_state=seed,
            normalize_y=True,
            copy_X_train=True,
            n_restarts_optimizer=n_restart,
            alpha=alpha,
        )
        gpr.fit(df_sd[var], df_sd[output])
        name = "GP ({})".format(str(gpr.kernel_))

        fun = FunctionGPR(gpr, var, [output], name, 0, var_min, var_max)
        functions.append(fun)

    ## Construct model
    return gr.Model(functions=functions, domain=domain, density=density)


ft_gp = add_pipe(fit_gp)

## Fit random forest model with sklearn
# --------------------------------------------------
@curry
def fit_rf(
    df,
    md=None,
    var=None,
    out=None,
    domain=None,
    density=None,
    seed=None,
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
        suppress_warnings (bool): Suppress warnings when fitting?

    Keyword Arguments:
        n_estimators (int):
        criterion (int):
        max_depth (int or None):
        min_samples_split (int, float):
        min_samples_leaf (int, float):
        min_weight_fraction_leaf (float):
        max_features (int, float, string):
        max_leaf_nodes (int or None):
        min_impurity_decrease (float):
        min_impurity_split (float):
        bootstrap (bool):
        oob_score (bool):
        n_jobs (int or None):
        random_state (int):

    Returns:
        gr.Model: A grama model with fitted function(s)

    Notes:
        - Wrapper for sklearn.ensemble.RandomForestRegressor

    """
    if suppress_warnings:
        filterwarnings("ignore")

    n_obs, n_in = df.shape

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
        rf = RandomForestRegressor(random_state=seed, **kwargs)
        rf.fit(df[var], df[output])
        name = "RF"

        fun = FunctionRegressor(rf, var, [output], name, 0)
        functions.append(fun)

    ## Construct model
    return gr.Model(functions=functions, domain=domain, density=density)


ft_rf = add_pipe(fit_rf)

## Fit linear model with sklearn
# --------------------------------------------------
@curry
def fit_lm(
    df,
    md=None,
    var=None,
    out=None,
    domain=None,
    density=None,
    seed=None,
    suppress_warnings=True,
    **kwargs
):
    r"""Fit a linear model

    Fit a linear model to given data. Specify inputs and outputs, or inherit
    from an existing model.

    Args:
        df (DataFrame): Data for function fitting
        md (gr.Model): Model from which to inherit metadata
        var (list(str) or None): List of features or None for all except outputs
        out (list(str)): List of outputs to fit
        domain (gr.Domain): Domain for new model
        density (gr.Density): Density for new model
        seed (int or None): Random seed for fitting process
        suppress_warnings (bool): Suppress warnings when fitting?

    Returns:
        gr.Model: A grama model with fitted function(s)

    Notes:
        - Wrapper for sklearn.ensemble.RandomForestRegressor

    """
    if suppress_warnings:
        filterwarnings("ignore")

    n_obs, n_in = df.shape

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
        lm = LinearRegression(**kwargs)
        lm.fit(df[var], df[output])
        name = "LM"

        fun = FunctionRegressor(lm, var, [output], name, 0)
        functions.append(fun)

    ## Construct model
    return gr.Model(functions=functions, domain=domain, density=density)


ft_lm = add_pipe(fit_lm)


## Fit kmeans clustering model
# --------------------------------------------------
@curry
def fit_kmeans(df, var=None, colname="cluster_id", seed=None, **kwargs):
    r"""K-means cluster a dataset

    Create a cluster-labeling model on a dataset using the K-means algorithm.

    Args:
        df (DataFrame): Hybrid point results from gr.eval_hybrid()
        var (list or None): Variables in df on which to cluster. Use None to
            cluster on all variables.
        colname (string): Name of cluster id; will be output in cluster model.
        seed (int): Random seed for kmeans clustering

    Kwargs:
        n_clusters (int): Number of clusters to fit
        random_state (int or None):

    Returns:
        gr.Model: Model that labels input data

    Notes:
        - A wrapper for sklearn.cluster.KMeans

    References:
        Scikit-learn: Machine Learning in Python, Pedregosa et al. JMLR 12, pp. 2825-2830, 2011.

    Examples:
        >>> import grama as gr
        >>> from grama.data import df_stang
        >>> from grama.fit import ft_kmeans
        >>> X = gr.Intention()
        >>> md_cluster = (
        >>>     df_stang
        >>>     >> ft_kmeans(var=["E", "mu"], n_clusters=2)
        >>> )
        >>> (
        >>>     md_cluster
        >>>     >> gr.ev_df(df_stang)
        >>>     >> gr.tf_group_by(X.cluster_id)
        >>>     >> gr.tf_summarize(
        >>>         thick_mean=gr.mean(X.thick),
        >>>         thick_sd=gr.sd(X.thick),
        >>>         n=gr.n(X.index),
        >>>     )
        >>> )

    """
    ## Check invariants
    if var is None:
        var = list(df.columns).copy()
    else:
        var = list(var).copy()
        diff = set(var).difference(set(df.columns))
        if len(diff) > 0:
            raise ValueError(
                "`var` must be subset of `df.columns`\n" "diff = {}".format(diff)
            )

    ## Generate clustering
    kmeans = KMeans(random_state=seed, **kwargs).fit(df[var].values)

    ## Build grama model
    def fun_cluster(df):
        res = kmeans.predict(df[var].values)
        return DataFrame(data={colname: res})

    md = gr.Model() >> gr.cp_vec_function(fun=fun_cluster, var=var, out=[colname])

    return md


ft_kmeans = add_pipe(fit_kmeans)
