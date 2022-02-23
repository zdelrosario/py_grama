from __future__ import division

import numpy as np

from .euclidean import EuclideanDomain

class RandomDomain(EuclideanDomain):
	r"""Abstract base class for domains with an associated sampling measure
	"""

	def pdf(self, X):
		r""" Probability density function associated with the domain

		This evaluates a probability density function :math:`p:\mathcal{D}\to \mathbb{R}_*`
		at the requested points. By definition, this density function is normalized
		to have measure over the domain to be one:

		.. math::

			\int_{\mathbf{x} \in \mathcal{D}} p(\mathbf{x}) \mathrm{d} \mathbf{x}.

		Parameters
		----------
		X: array-like, either (m,) or (N,m)
			points to evaluate the density function at
	
		Returns
		-------
		array-like (N,)
			evaluation of the density function

		"""
		X = np.atleast_2d(np.array(X))
		return self._pdf(X)

	def _pdf(self, x):
		raise NotImplementedError
