import numpy as np
import cvxpy as cp


def miniball(X, L = None):
	r""" Compute the smallest enclosing sphere

	TODO: Use implementation of https://github.com/hbf/miniball

	Parameters
	----------
	X: array-like (M,m)
		Points to enclose in a ball
	L: optional, (m,m)
		Lipschitz-like weighting metric
	
	Returns
	-------
	x: np.array(m)
		Center of circle
	r: float
		radius of circle
	"""
	X = np.array(X)
	M, m = X.shape
	if L is None:
		L = np.eye(m)

	x = cp.Variable(m)
	ones = np.ones((1, M))
	obj = cp.mixed_norm( (L @ ( cp.reshape(x,(m,1)) @ ones - X.T)).T, 2, np.inf)
	prob = cp.Problem(cp.Minimize(obj))
	prob.solve()

	return x.value, obj.value
