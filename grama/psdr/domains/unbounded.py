from __future__ import division

import numpy as np

from .euclidean import EuclideanDomain

class UnboundedDomain(EuclideanDomain):
	r""" A domain without any constraints
	
	This class implements a subset of the functionality of the Domain
	applicable for a domain that is all of :math:`\mathbb{R}^m`.

	Parameters
	----------
	dimension: int
		Number of unbounded dimensions
	names: list of strings
		Names for each dimension

	"""
	def __init__(self, dimension, names = None):
		self._dimension = dimension
		self._init_names(names)

		self._unbounded = True
		self._empty = False
		self._point = False

	def __str__(self):
		return u"<UnboundedDomain on R^%d>" % (len(self),)
	
	def __len__(self):
		return self._dimension

	def _build_constraints(self, x):
		return [] 
	
	def _build_constraints_norm(self, x_norm):
		return []

	def _closest_point(self, x0, L = None, **kwargs):
		return x0

	def _normalize(self, X):
		return X

	def _unnormalize(self, X_norm):
		return X_norm

	def _isinside(self, X, tol = None):
		if X.shape[1]== len(self):
			return np.ones(X.shape[0],dtype = np.bool)
		else:
			return np.zeros(X.shape[0],dtype = np.bool)

