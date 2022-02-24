all = [
    "eval_pnd",
    "ev_pnd",
    "pareto_min_rel",
    "make_proposal_sigma",
    "rprop",
    "dprop",
    "approx_pnd"
]

from grama import add_pipe, Model, tf_md

from numpy import array, diag, dot, ones, mean, zeros
from numpy import any as npany
from numpy import all as npall
from numpy import min as npmin
from numpy import max as npmax
from numpy import sum as npsum
from numpy.linalg import norm
from numpy.random import choice, multivariate_normal
from numpy.random import seed as set_seed
from pandas import DataFrame
from scipy.linalg import svd
from scipy.stats import multivariate_normal as mvnorm
from toolz import curry


@curry
def eval_pnd(model, df_train, df_test, signs, n=int(1e4), seed=None, append=True, \
    mean_prefix="_mean", sd_prefix="_sd"):
    """ Evaluate a Model using a predictive model

    Evaluates a given model against a PND algorithm to determine
    "optimal points".

    Args:
        model (gr.model): predictive model to evaluate
        df_train (DataFrame): dataframe with training data
        df_test (DataFrame): dataframe with test data
        signs (dict): dict with the variables you would like to use and
            minimization or maximization parameter for each
        append (bool): Append df_test to pnd algorithm outputs

    Kwargs:
        n (int): Number of draws for importance sampler
        seed (int): declarble seed value for reproducibility

    Returns:
        DataFrame: Results of predictive model going through a PND algorithm.
        Conatians both values and their scores.

    Example:
    >>> import grama as gr
    >>>
    >>> md_true = gr.make_pareto_random()
    >>>
    >>> df_data = (
    >>>     md_true
    >>>     >> gr.ev_sample(n=2e3, seed=101, df_det="nom")
    >>> )
    >>>
    >>> df_train = (
    >>>     df_data
    >>>     >> gr.tf_sample(n=10))
    >>> )
    >>>
    >>> df_test = (
    >>>     df_data
    >>>     >> gr.anti_join(
    >>>         df_train,
    >>>         by = ["x1","x2"]
    >>>     )
    >>>     >> gr.tf_sample(n=200)
    >>> )
    >>>
    >>> md_fit = (
    >>>     df_train
    >>>     >> gr.ft_gp(
    >>>         var=["x1","x2"]
    >>>         out=["y1","y2"]
    >>>     )
    >>> )
    >>>
    >>> df_pnd = (
            md_fit
            >> gr.ev_pnd(
                df_train,
                df_test,
                signs = {"y1":1, "y2":1},
                seed = 101
            )
            >> gr.tf_arrange(gr.desc(DF.pr_scores))
        )
    """
    # # Check for correct types
    # if not isinstance(model, Model):
    #     raise TypeError('model must be a Model')

    # if not isinstance(df_train, DataFrame):
    #     raise TypeError('df_train must be a DataFrame')
    #
    # if not isinstance(df_test, DataFrame):
    #     raise TypeError('df_test must be a DataFrame')

    # Check content
    if len(model.out)/2 < 2:
        raise ValueError('Given Model needs multiple outputs')

    if len(model.functions) == 0:
        raise ValueError("Given model has no functions")

    if not set(model.var).issubset(set(df_train.columns)):
        raise ValueError("model.var must be subset of df_train.columns")

    if not set(model.var).issubset(set(df_test.columns)):
        raise ValueError("model.var must be subset of df_test.columns")

    for key in signs.keys():
        if key+mean_prefix not in model.out:
            raise ValueError(f"signs.{key} implies output {key+mean_prefix}, which is not found in provided md.out")
        if key+sd_prefix not in model.out:
            raise ValueError(f"signs{key} implies output {key+sd_prefix}, which is not found in provided sd.out")

    ## Compute predictions and predicted uncertainties
    df_pred = (
        df_test
        >> tf_md(md=model)
    )

    ## Setup for reshaping
    means = []
    sds = []
    columns = df_train.columns.values
    length = int(len(signs.keys()))
    outputs = [key for key in signs.keys() if key in columns]
    signs = [value for value in signs.values()]

    ## append mean and sd prefixes
    for i, value in enumerate(outputs):
        means.append(value+mean_prefix)
        sds.append(value+sd_prefix)

    ## Remove extra columns from df_test
    df_pred = df_pred[means + sds]

    ## Reshape data for PND algorithm
    X_pred = df_pred[means].values      # Predicted response values
    X_sig = df_pred[sds].values         # Predictive uncertainties
    X_train = df_train[outputs].values  # Training

    ### Create covariance matrices
    X_cov = zeros((X_sig.shape[0], length, length))
    for l in range(length):
        for i in range(X_sig.shape[0]):
            X_cov[i, l, l] = X_sig[i, l]

    ### Apply pnd
    pr_scores, var_values = approx_pnd(
        X_pred,
        X_cov,
        X_train,
        signs = signs,
        n = n,
        seed = seed
    )

    ### Package outputs
    df_pnd = DataFrame(
        {
            "pr_scores": pr_scores,
            "var_values": var_values,
        }
    )

    if append:
        return df_test.reset_index(drop=True).merge(df_pnd, left_index=True, right_index=True)
    return df_pnd

ev_pnd = add_pipe(eval_pnd)


# Relative Pareto frontier calculation
def pareto_min_rel(X_test, X_base=None):
    r"""Determine if rows in X_test are optimal, compared to X_base

    Finds the Pareto-efficient test-points that minimize the column values,
    relative to a given set of base-points.

    Args:
        X_test (2d numpy array): Test point observations; rows are observations, columns are features
        X_base (2d numpy array): Base point observations; rows are observations, columns are features

    Returns:
        array of boolean values: Indicates if test observation is Pareto-efficient, relative to base points

    References:
        Owen *Monte Carlo theory, methods and examples* (2013)
    """
    # Compute Pareto points
    is_efficient = ones(X_test.shape[0], dtype=bool)

    if X_base is None:
        for i, x in enumerate(X_test):
            is_efficient[i] = npall(npany(X_test[:i] > x, axis=1)) and npall(
                npany(X_test[i + 1 :] > x, axis=1)
            )
    else:
        for i, x in enumerate(X_test):
            is_efficient[i] = not (
                npany(npall(x >= X_base, axis=1))
                and npany(npany(x >  X_base, axis=1))
            )

    return is_efficient

# Floor an array of variances
def floor_sig(sig, sig_min=1e-8):
    r"""Floor an array of variances
    """
    sig_floor = npmin([norm(sig), sig_min])
    return list(map(lambda s: npmax([s, sig_floor]), sig))

# Estimate parameters for PND importance distribution
def make_proposal_sigma(X, idx_pareto, X_cov):
    r"""Estimate parameters for PND importance distribution

    Args:
        X (2d numpy array): Full set of response values
        idx_pareto (boolean array): Whether each response value is a Pareto point
        X_cov (iterable of 2d numpy array): Predictive covariance matrices

    Preconditions:
        X.shape[0] == len(idx_pareto)
        X_cov[i].shape[0] == X.shape[1] for all valid i
        X_cov[i].shape[1] == X.shape[1] for all valid i

    Returns:
        2d numpy array: Common covariance matrix for PND importance distribution
    """
    n_pareto = len(idx_pareto)
    n_dim = X.shape[0]
    X_mean = mean(X[idx_pareto, :], axis=0)

    ## If sufficient, use Pareto points only
    if n_pareto > n_dim:
        U, sig, Vh = svd(X[idx_pareto, :] - X_mean)
    ## If insufficient, use full distribution
    else:
        U, sig, Vh = svd(X - X_mean)

    ## Find largest predictive covariance component to avoid dangerously light tails
    sig_min = npmax(X_cov)
    ## Apply heuristic based on Owen "Monte Carlo", Exercise 9.7
    sig_min = ((5/4)**1/2) * sig_min

    ## Floor the variances
    sig = floor_sig(sig, sig_min=sig_min)

    ## TODO: Develop better heuristic
    Sigma = dot(Vh.T, dot(diag(sig), Vh)) / n_pareto

    return Sigma

# Draw from mixture distribution
def rprop(n, Sigma, X_means, seed=None):
    r"""Draw a sample from gaussian mixture distribution

    Draw a sample from a gaussian mixture distribution with a common covariance
    matrix and different means.

    Args:
        n (int): Number of observations in sample
        Sigma (2d numpy array): Common covariance matrix for mixture distribution
        X_means (2d numpy array): Means for mixture distribution

    Preconditions:
        Sigma.shape[0] == X_means.shape[1]
        Sigma.shape[1] == X_means.shape[1]

    Returns:
        2d numpy array: Sample of observations
    """
    if seed is not None:
        set_seed(seed)

    n_pareto, n_dim = X_means.shape
    X_sample = zeros((n, n_dim))

    for i in range(n):
        idx = choice(n_pareto)
        X_sample[i, :] = multivariate_normal(
            mean=X_means[idx, :],
            cov=Sigma,
            size=1,
        )

    return X_sample

# Calculate density based on proposal distribution
def dprop(X, Sigma, X_means):
    r"""Evaluate the PDF of a mixture proposal distribution

    Evaluate the PDF of a gaussian mixture distribution with a common covariance
    matrix and different means.

    Args:
        X (2d numpy array): Observations for which to evaluate the density
        Sigma (2d numpy array): Common covariance matrix for mixture distribution
        X_means (2d numpy array): Means for mixture distribution

    Preconditions:
        X.shape[1] == X_means.shape[1]
        Sigma.shape[0] == X_means.shape[1]
        Sigma.shape[1] == X_means.shape[1]

    Returns:
        numpy array: Density values
    """
    n, n_dim = X.shape
    n_comp = X_means.shape[0]
    w = 1 / n_comp           # Equal weighting of mixture components

    L = zeros((n_comp, n))
    dist = mvnorm(cov=Sigma)
    for i in range(n_comp):
        L[i, :] = dist.pdf(X - X_means[i])

    return npsum(L, axis=0) * w

# Approximate PND via IS
def approx_pnd(X_pred, X_cov, X_train, signs, n=int(1e4), seed=None):
    r"""Approximate the PND via mixture importance sampling

    Approximate the probability non-dominated (PND) for a set of predictive
    points using a mixture importance sampling approach. Predictive points are
    assumed to have predictive gaussian distributions (with specified mean and
    covariance matrix).

    Args:
        X_pred (2d numpy array): Predictive values
        X_cov (iterable of 2d numpy arrays): Predictive covariance matrices
        X_train (2d numpy array): Training values, used to determine existing Pareto frontier
        signs (numpy array of +/-1 values): Array of optimization signs: {-1: Minimize, +1 Maximize}

    Kwargs:
        n (int): Number of draws for importance sampler
        seed (int): Seed for random state

    Returns:
        pr_scores (array): Estimated PND values
        var_values (array): Estimated variance values

    References:
        Owen *Monte Carlo theory, methods and examples* (2013)

    """
    ## Setup
    X_wk_train = -X_train * signs
    X_wk_pred = -X_pred * signs
    n_train, n_dim = X_train.shape
    n_pred = X_pred.shape[0]

    ## Find the training Pareto frontier
    idx_pareto = pareto_min_rel(X_wk_train)
    n_pareto = len(idx_pareto)
    ## Sample the mixture points
    Sig_mix = make_proposal_sigma(X_wk_train, idx_pareto, X_cov)
    X_mix = rprop(n, Sig_mix, X_wk_train[idx_pareto, :], seed=seed)
    ## Take non-dominated points only
    idx_ndom = pareto_min_rel(X_mix, X_base=X_wk_train[idx_pareto, :])
    X_mix = X_mix[idx_ndom, :]

    ## Evaluate the Pr[non-dominated]
    d_mix = dprop(X_mix, Sig_mix, X_wk_train[idx_pareto, :])
    pr_scores = zeros(n_pred)
    var_values = zeros(n_pred)
    for i in range(n_pred):
        dist_test = mvnorm(mean=X_wk_pred[i], cov=X_cov[i])
        w_test = dist_test.pdf(X_mix) / d_mix
        # Owen (2013), Equation (9.3)
        pr_scores[i] = npsum(w_test) / n
        # Owen (2013), Equation (9.5)
        var_values[i] = npsum( (w_test - pr_scores[i])**2 ) / n

    return pr_scores, var_values
