from __future__ import division

import numpy as np
from .domain import TOL
from .box import BoxDomain
from .random import RandomDomain
from .normal import NormalDomain

# TODO: Ensure sampling is still correct (IMPORTANT FOR DUU Solution)
class LogNormalDomain(BoxDomain, RandomDomain):
	r"""A one-dimensional domain described by a log-normal distribution.

	Given a normal distribution :math:`\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Gamma})`,
	the log normal is described by

	.. math::

		x = \alpha + \beta e^y, \quad y \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Gamma})

	where :math:`\alpha` is an offset and :math:`\beta` is a scaling coefficient. 
	

	Parameters
	----------
	mean: float
		mean of normal distribution feeding the log-normal distribution
	cov: float, optional  
		covariance of normal distribution feeding the log-normal distribution
	offset: float, optional
		Shift the distribution
	scaling: float or np.ndarray
		Scale the output of the log-normal distribution
	truncate: float [0,1)
		Truncate the tails of the distribution
	"""	
	def __init__(self, mean, cov = 1., offset = 0., scaling = 1., truncate = None, names = None):
		self.tol = 1e-6
		self.normal_domain = NormalDomain(mean, cov, truncate = truncate)
		assert len(self.normal_domain) == 1, "Only defined for one-dimensional distributions"

		self.mean = float(self.normal_domain.mean)
		self.cov = float(self.normal_domain.cov)
		self.scaling = float(scaling)
		self.offset = float(offset)
		self.truncate = truncate

		# Determine bounds
		# Because this doesn't have a convex relationship to the multivariate normal
		# truncated domains, we manually specify these here as they cannot be inferred
		# from the (non-existant) quadratic constraints as in the NormalDomain case.
		if self.truncate is not None:
			self._lb = self.offset + self.scaling*np.exp(self.normal_domain.norm_lb)
			self._ub = self.offset + self.scaling*np.exp(self.normal_domain.norm_ub)
		else:
			self._lb = 0.*np.ones(1)
			self._ub = np.inf*np.ones(1)

		self._init_names(names)

	def __len__(self):
		return len(self.normal_domain)

	def _sample(self, draw = 1):
		X = self.normal_domain.sample(draw)
		return np.array(self.offset).reshape(-1,1) + self.scaling*np.exp(X)

	def _normalized_domain(self, **kwargs):
		names_norm = [name + ' (normalized)' for name in self.names]
		if self.truncate is not None:
			c = self._center()
			D = float(self._normalize_der()) 
			return LogNormalDomain(self.normal_domain.mean, self.normal_domain.cov, 
				offset = D*(self.offset - c) , scaling = D*self.scaling, truncate = self.truncate, names = names_norm)
		else:
			return self

	def _pdf(self, X):
		X_norm = (X - self.offset)/self.scaling
		p = np.exp(-(np.log(X_norm) - self.mean)**2/(2*self.cov))/(X_norm*self.cov*np.sqrt(2*np.pi))
		return p
