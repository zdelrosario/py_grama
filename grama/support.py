# Implementation of support points generator
#
# Reference:
#   Mak and Joseph, "Support Points" (2018) *The Annals of Statistics*

from numpy import ma, newaxis
from numpy.linalg import norm
from numpy.random import choice


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
        # TODO: Parallel for
        for i in range(n):
            Xn[i] = _iterate_x(X0, Y, i)

        # Update loop variables
        d = norm(X0 - Xn, axis=1).mean()
        iter_c = iter_c + 1

        # Overwrite X0
        X0 = Xn.copy()

    ## DEBUG
    return Xn, d, iter_c


## DEBUG test
if __name__ == "__main__":
    import numpy as np

    X0 = np.random.multivariate_normal([0, 0], np.eye(2), size=10)
    Y = np.random.multivariate_normal([0, 0], np.array([[1, 0.5], [0.5, 1]]), size=100)

    #
    np.random.seed(101)
    X, d, iter_c = _sp_cpp(X0, Y)
