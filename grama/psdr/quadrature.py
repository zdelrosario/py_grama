from __future__ import print_function
import numpy as np
import scipy.linalg

__all__ = ['gauss']

def gauss(N, a = 0, b = 1):
	r""" Gauss-Legendre quadrature rule 



	Parameters
	----------
	N: int
		number of samples to use in quadrature rule
	a: float, default 0
		left endpoint
	b: float, default 1
		right endpoint

	Returns
	-------
	x: np.ndarray
		locations at which to sample
	w: np.ndarray
		weights

	Notes
	-----
	This code is adapted from Mark Embree's `gaussab.m`,
	a modification of `gauss.m` from Trefethen's Spectral Methods in Matlab
	"""
	a, b = float(a), float(b)

	beta = 0.5/np.sqrt(1 - ((np.arange(2,2*N, 2, dtype = np.float))**(-2) ))
	T = np.diag(beta, 1) + np.diag(beta, -1)

	# Because eigenvalues come sorted in increasing order, there is no need to sort
	ew, ev = scipy.linalg.eigh(T)
	x = ew
	w = 2*ev[0,:]**2

	# Mark's modifications
	x = a + (b - a)/2.*(1.+x)
	w = ((b-a)/2.)*w

	return x, w

