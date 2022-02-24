import numpy as np
import scipy.spatial.distance

def cdist(X1, X2, L = None):
	r""" A convience wrapper around a weighted l2-norm distance calculation
	"""
	if len(X1.shape) == 1:
		X1 = np.reshape(X1,(1,-1))
	if len(X2.shape) == 1:
		X2 = np.reshape(X2,(1,-1))

	if L is not None:
		X1 = (L @ X1.T).T
		X2 = (L @ X2.T).T

	return scipy.spatial.distance.cdist(X1, X2)
