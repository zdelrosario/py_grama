__all__ = ["fit_gp", "ft_gp"]

## Fitting via sklearn package
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Con
except ModuleNotFoundError:
    raise ModuleNotFoundError("module sklearn not found")

import grama as gr

from grama import pipe
from pandas import DataFrame
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
    def __init__(self, gpr, df_train, var, out, name, runtime):
        self.gpr = gpr
        self.df_train = df_train
        self.var = var
        self.out = out
        self.name = name
        self.runtime = runtime

        self.ser_min_in = df_train[var].min()
        self.ser_max_in = df_train[var].max()
        self.ser_min_out = df_train[out].min()
        self.ser_max_out = df_train[out].max()

    def eval(self, df):
        ## Check invariant; model inputs must be subset of df columns
        if not set(self.var).issubset(set(df.columns)):
            raise ValueError(
                "Model function `{}` var not a subset of given columns".format(
                    self.name
                )
            )
        df_std = standardize_cols(df, self.ser_min_in, self.ser_max_in, self.var)
        y = self.gpr.predict(df_std[self.var])
        return restore_cols(
            DataFrame(data=y, columns=self.out),
            self.ser_min_out,
            self.ser_max_out,
            self.out,
        )

    def copy(self):
        func_new = FunctionGPR(
            self.gpr, self.df_train.copy(), self.var, self.out, self.name, self.runtime
        )

        return func_new


## Fit GP model with sklearn
# --------------------------------------------------
@curry
def fit_gp(
    df,
    md=None,
    inputs=None,
    outputs=None,
    domain=None,
    density=None,
    kernel=None,
    seed=None,
    suppress_warnings=True,
    n_restart=5,
    alpha=1e-10,
):
    r"""Fit a gaussian process

    Fit a gaussian process to given data. Specify inputs and outputs, or inherit
    from an existing model.

    Args:
        df (DataFrame): Data for function fitting
        md (gr.Model): Model from which to inherit metadata
        inputs (list(str) or None): List of features or None for all except outputs
        outputs (list(str)): List of outputs to fit
        domain (gr.Domain): Domain for new model
        density (gr.Density): Density for new model
        seed (int or None): Random seed for fitting process
        kernel (sklearn.gaussian_process.kernels.Kernel or None): Kernel for GP
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
        outputs = md.out

    ## Check invariants
    if not set(outputs).issubset(set(df.columns)):
        raise ValueError("outputs must be subset of df.columns")
    ## Default input value
    if inputs is None:
        inputs = list(set(df.columns).difference(set(outputs)))
    ## Check more invariants
    set_inter = set(outputs).intersection(set(inputs))
    if len(set_inter) > 0:
        raise ValueError(
            "outputs and inputs must be disjoint; intersect = {}".format(set_inter)
        )
    if not set(inputs).issubset(set(df.columns)):
        raise ValueError("inputs must be subset of df.columns")

    ## Pre-process data
    ser_min_in = df[inputs].min()
    ser_max_in = df[inputs].max()
    ser_min_out = df[outputs].min()
    ser_max_out = df[outputs].max()
    df_std = standardize_cols(df, ser_min_in, ser_max_in, inputs)
    df_std = standardize_cols(df_std, ser_min_out, ser_max_out, outputs)

    ## Assign default kernel, if necessary
    if kernel is None:
        print("fit_gp is assigning default kernel")
        kernel = Con(1, (1e-3, 1e3)) * RBF([1] * len(inputs), (1e-8, 1e8))

    ## Construct gaussian process for each output
    functions = []

    for output in outputs:
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=seed,
            normalize_y=False,
            copy_X_train=False,
            n_restarts_optimizer=n_restart,
            alpha=alpha,
        )
        gpr.fit(df_std[inputs], df_std[output])
        name = "GP ({})".format(str(gpr.kernel_))

        fun = FunctionGPR(gpr, df.copy(), inputs, [output], name, 0)
        functions.append(fun)

    ## Construct model
    return gr.Model(functions=functions, domain=domain, density=density)


@pipe
def ft_gp(*args, **kwargs):
    return fit_gp(*args, **kwargs)
