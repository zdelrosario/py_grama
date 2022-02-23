from __future__ import division

import numpy as np

from .domain import TOL
from .box import BoxDomain

class PointDomain(BoxDomain):
	r""" A domain consisting of a single point

	Given a point :math:`\mathbf{x} \in \mathbb{R}^m`, construct the domain consisting of that point

	.. math::
	
		\mathcal{D} = \lbrace \mathbf x \rbrace \subset \mathbb{R}^m.

	Parameters
	----------
	x: array-like (m,)
		Single point to be contained in the domain
	"""
	def __init__(self, x, names = None):
		self._point = True
		self._empty = False
		self._unbounded = False
		self._x = np.array(x).flatten()
		assert len(self._x.shape) == 1, "Must provide a one-dimensional point"
		
		BoxDomain.__init__(self, lb = self._x, ub = self._x, names = names)

	def __len__(self):
		return self._x.shape[0]
		
	def _closest_point(self, x0, **kwargs):
		return np.copy(self._x)

	def _corner(self, p, **kwargs):
		return np.copy(self._x)

	def _extent(self, x, p, **kwargs):
		return 0

	def _isinside(self, X, tol = TOL):
		Pcopy = np.tile(self._x.reshape(1,-1), (X.shape[0],1))
		return np.all(X == Pcopy, axis = 1)	

	def _sample(self, draw = 1):
		return np.tile(self._x.reshape(1,-1), (draw, 1))


#	@property
#	def lb(self):
#		return np.copy(self._x)
#
#	@property
#	def ub(self):
#		return np.copy(self._x)
