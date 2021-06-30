# Implementation of support points generator
#
# Reference:
#   Mak and Joseph, "Support Points" (2018) *The Annals of Statistics*

__all__ = [
    "tran_sp",
    "tf_sp",
]

from grama import add_pipe
from numpy import diag, eye, ma, newaxis, number, zeros
from numpy.linalg import norm
from numpy.random import choice, multivariate_normal
from numpy.random import seed as setseed
from pandas import DataFrame
from toolz import curry
from warnings import warn


## Helper functions
##################################################
def _iterate_x(X, Y, ind):
    r"""Iterate a single candidate point

    Implementation of Equation (22) from Mak and Joseph (2018)

    Arguments:
        X (np.array): candidate points, X.shape == (n, p)
        Y (np.array): target points, Y.shape == (N, p)
        ind (int): candidate to iterate, 0 <= ind <= n - 1

    Returns:
        np.array: updated candidate point
    """
    ## Setup
    n = X.shape[0]
    N = Y.shape[0]

    ## Compute iteration
    # First term
    diffx = ma.array(X[ind] - X, mask=False)
    diffx[ind].mask = True
    diffx_norm = ma.array(norm(diffx, axis=1), mask=False)
    diffx_norm.mask[ind] = True
    t1 = (N / n) * (diffx / diffx_norm[:, newaxis]).sum(axis=0)

    # Second term
    diffy_norm = norm(X[ind] - Y, axis=1)
    q = (1 / diffy_norm).sum()
    t2 = (Y / diffy_norm[:, newaxis]).sum(axis=0)

    return (1 / q) * (t1 + t2)


def _sp_cpp(X0, Y, delta=1e-6, iter_max=500):
    r"""Implementation of sp.cpp algorithm

    Implementation of sp.cpp algorithm from Mak and Joseph (2018). Note that
    this implementation takes

    Signature:
        X, d, iter_c = _sp_cpp(X0, Y)

    Arguments:
        X0 (np.array): initial candidate points, X0.shape == (n, p)
        Y (np.array): target points, Y.shape == (N, p)
        delta (float): convergence criterion, as average pairwise-distance
            between iterations
        iter_max (int): maximum iteration count

    Returns:
        X (np.array): optimized support points
        d (float): average pairwise-distance at termination
        iter_c (int): iteration count at termination

    """
    ## Setup
    N, p = Y.shape
    n = X0.shape[0]
    Xn = X0.copy()

    ## Primary loop
    d = delta * 2
    iter_c = 0
    # Check convergence criterion
    while (d >= delta) and (iter_c < iter_max):
        # Update the candidate points
        for i in range(n):
            Xn[i] = _iterate_x(X0, Y, i)

        # Update loop variables
        d = norm(X0 - Xn, axis=1).mean()
        iter_c = iter_c + 1

        # Overwrite X0
        X0 = Xn.copy()

    ## DEBUG
    return Xn, d, iter_c


def _perturbed_choice(Y, n):
    r"""Choose a set of perturbed points

    Arguments:
        Y (np.array): target points, Y.shape == (N, p)

    Returns:
        np.array: perturbed points, shape == (n, p)

    """
    i0 = choice(Y.shape[0], size=n)
    # Add noise to initial proposal to avoid X-Y overlap;
    # random directions with fixed distance
    V_rand = multivariate_normal(zeros(Y.shape[1]), eye(Y.shape[1]), size=n)
    V_rand = V_rand / norm(V_rand, axis=1)[:, newaxis]
    X0 = Y[i0] + V_rand * Y.std(axis=0)

    return X0


## Public interfaces
##################################################
@curry
def tran_sp(
    df,
    n=None,
    var=None,
    n_maxiter=500,
    tol=1e-3,
    seed=None,
    verbose=True,
    standardize=True,
):
    r"""Compact a dataset with support points

    Arguments:
        df (DataFrame): dataset to compact
        n (int): number of samples for compacted dataset
        var (list of str): list of variables to compact, must all be numeric
        n_maxiter (int): maximum number of iterations for support point algorithm
        tol (float): convergence tolerance
        verbose (bool): print messages to the console?
        standardize (bool): standardize columns before running sp? (Restores after sp)

    Returns:
        DataFrame: dataset compacted with support points

    Examples:
        >>> import grama as gr
        >>> from grama.data import df_diamonds
        >>> df_sp = gr.tran_sp(df_diamonds, n=50, var=["price", "carat"])
    """
    ## Setup
    setseed(seed)
    # Handle input variables
    if var is None:
        # Select numeric columns only
        var = list(df.select_dtypes(include=[number]).columns)
        if verbose:
            print("tran_sp has selected var = {}".format(var))
    # Extract values
    Y = df[var].values
    if standardize:
        Y_mean = Y.mean(axis=0)
        Y_sd = Y.std(axis=0)
        Y = (Y - Y_mean) / Y_sd
    # Generate initial proposal points
    X0 = _perturbed_choice(Y, n)

    ## Run sp.ccp algorithm
    X, d, iter_c = _sp_cpp(X0, Y, delta=tol, iter_max=n_maxiter)
    if verbose:
        print(
            "tran_sp finished in {0:} iterations with distance criterion {1:4.3e}".format(
                iter_c, d
            )
        )
        if d > tol:
            warn(
                "Convergence tolerance not met; d = {0:4.3e} > tol = {1:4.3e}".format(
                    d, tol
                ),
                RuntimeWarning,
            )

    if standardize:
        X = X * Y_sd + Y_mean

    ## Package results
    return DataFrame(data=X, columns=var)


tf_sp = add_pipe(tran_sp)
