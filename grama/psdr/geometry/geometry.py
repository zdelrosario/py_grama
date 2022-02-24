""" Utility containing various geometric routines, most of which are used in sampling
"""
from __future__ import print_function
import numpy as np
from scipy.spatial import Voronoi 
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = ['sample_sphere', 'unique_points', 'sample_simplex']




def sample_sphere(dim, n, k = 100):
	""" Sample points on a high-dimensional sphere 

	Uses Mitchell's best candidate algorithm to obtain a 
	quasi-uniform distribution of points on the sphere,


	See:
		https://www.jasondavies.com/maps/random-points/
		https://bl.ocks.org/mbostock/d7bf3bd67d00ed79695b

	Parameters
	----------
	dim: int, positive
		Dimension of the space to sample
	n: int, positive
		Number of points to sample
	k: int, positive (optional)
		Number of candidates to take at each step
	"""
	X = np.zeros( (n, dim) )
	
	# First sample
	x = np.random.randn(dim)
	x /= np.linalg.norm(x)
	X[0] = x
	
	for i in range(1,n):
		# Draw candidates (normalized points on the sphere)
		Xcan = np.random.randn(k, dim)
		Xcan = (Xcan.T/np.sqrt(np.sum(Xcan**2, axis = 1))).T

		# Compute the distance
		dist = np.min(1 - np.dot(X[:i,], Xcan.T), axis = 0)
		I = np.argmax(dist)
		X[i] = Xcan[I]

	return X

def unique_points(X):
	r""" Compute the unique points from a list

	Parameters
	----------
	X: array-like (M, m)
		Input points
	
	Returns
	-------
	I: np.ndarray (M, dtype = np.bool)
		List of points with no points close to each other
	"""
	D = squareform(pdist(X))
	I = np.ones(len(X), dtype = np.bool)
	for i in range(len(X)-1):
		I[i] = ~np.isclose(np.min(D[i,i+1:]), 0)
	return I


def sample_simplex(dim, Nsamp = 1):
	r""" Samples the unit simplex uniformly



	References:
	https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex

	"""
	r = np.random.uniform(0, 1, size = (Nsamp, dim+1))
	r[:,0] = 0
	r[:,-1] = 1
	r = np.sort(r, axis = 1)
	alphas = r[:,1:] - r[:,:-1]
	return alphas
