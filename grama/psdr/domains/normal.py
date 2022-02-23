from __future__ import division

import numpy as np
import scipy

from ..misc import merge
from .domain import TOL, DEFAULT_CVXPY_KWARGS
from .random import RandomDomain
from .linquad import LinQuadDomain

class NormalDomain(LinQuadDomain, RandomDomain):
	r""" Domain described by a normal distribution

	This class describes a normal distribution with 
	mean :math:`\boldsymbol{\mu}\in \mathbb{R}^m` and 
	a symmetric positive definite covariance matrix :math:`\boldsymbol{\Gamma}\in \mathbb{R}^{m\times m}`
	that has the probability density function:

	.. math:: 

		p(\mathbf{x}) = \frac{
			e^{-\frac12 (\mathbf{x} - \boldsymbol{\mu}) \boldsymbol{\Gamma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}
			}{\sqrt{(2\pi)^m |\boldsymbol{\Gamma}|}} 

	If the parameter :code:`truncate` is specified, this distribution is truncated uniformly; i.e.,
	calling this parameter :math:`\tau`, the resulting domain has measure :math:`1-\tau`.
	Specifically, if we have a Cholesky decomposition of :math:`\boldsymbol{\Gamma} = \mathbf{L} \mathbf{L}^\top`
	we find a :math:`\rho` such that

	.. math::
		
		\mathcal{D} &= \lbrace \mathbf{x}: \|\mathbf{L}^{-1}(\mathbf{x} - \boldsymbol{\mu})\|_2^2 \le \rho\rbrace ; \\
		p(\mathcal{D}) &= 1-\tau.
		 
	This is done so that the domain has compact support which is necessary for several metric-based sampling strategies.


	Parameters
	----------
	mean : array-like (m,)
		Mean 
	cov : array-like (m,m), optional
		Positive definite Covariance matrix; defaults to the identity matrix
	truncate: float in [0,1), optional
		Amount to truncate the domain to ensure compact support
	"""
	def __init__(self, mean, cov = None, truncate = None, names = None, **kwargs):
		self.tol = 1e-6	
		self.kwargs = merge(DEFAULT_CVXPY_KWARGS, kwargs)
		######################################################################################	
		# Process the mean
		######################################################################################	
		if isinstance(mean, (int, float)):
			mean = [mean]
		self.mean = np.array(mean)
		self._dimension = m = self.mean.shape[0]
	
		######################################################################################	
		# Process the covariance
		######################################################################################	
		if isinstance(cov, (int, float)):
			cov = np.array([[cov]])
		elif cov is None:
			cov = np.eye(m)
		else:
			cov = np.array(cov)
			assert cov.shape[0] == cov.shape[1], "Covariance must be square"
			assert cov.shape[0] == len(self),  "Covariance must be the same shape as mean"
			assert np.all(np.isclose(cov,cov.T)), "Covariance matrix must be symmetric"
		
		self.cov = cov
		self.L = scipy.linalg.cholesky(self.cov, lower = True)		
		self.Linv = scipy.linalg.solve_triangular(self.L, np.eye(len(self)), lower = True, trans = 'N')
		self.truncate = truncate

		if truncate is not None:
			# Clip corresponds to the 2-norm squared where we should trim based on the truncate
			# parameter.  1 - cdf is the survival function, so we call the inverse survival function to locate
			# this parameter.
			self.clip = scipy.stats.chi2.isf(truncate, len(self)) 
			
			self._Ls = [np.copy(self.Linv) ]
			self._ys = [np.copy(self.mean)]
			# As the transform by Linv places this as a standard-normal,
			# we truncate at the clip parameter.
			self._rhos = [np.sqrt(self.clip)]

		else:
			self.clip = None
			self._Ls = []
			self._ys = []
			self._rhos = []

		self._init_names(names)

	def _sample(self, draw = 1):
		X = np.random.randn(draw, self.mean.shape[0])
		if self.clip is not None:
			# Under the assumption that the truncation parameter is small,
			# we use replacement sampling.
			while True:
				# Find points that violate the clipping
				I = np.sum(X**2, axis = 1) > self.clip
				if np.sum(I) == 0:
					break
				X[I,:] = np.random.randn(np.sum(I), self.mean.shape[0])
		
		# Convert from standard normal into this domain
		X = (self.mean.reshape(-1,1) + self.L.dot(X.T) ).T
		return X


	def _center(self):
		# We redefine the center because due to anisotropy in the covariance matrix,
		# the center is *not* the mean of the coordinate-wise bounds
		return np.copy(self.mean)

	def _normalized_domain(self, **kwargs):
		# We need to do this to keep the sampling measure correct
		names_norm = [name + ' (normalized)' for name in self.names]
		D = self._normalize_der()
		return NormalDomain(self.normalize(self.mean), D.dot(self.cov).dot(D.T), truncate = self.truncate, names = names_norm, 
			**merge(self.kwargs, kwargs))

	

	################################################################################		
	# Simple properties
	################################################################################		
	@property
	def lb(self): return -np.inf*np.ones(len(self))
	
	@property
	def ub(self): return np.inf*np.ones(len(self))

	@property
	def A(self): return np.zeros((0,len(self)))

	@property
	def b(self): return np.zeros(0)

	@property
	def A_eq(self): return np.zeros((0,len(self)))

	@property
	def b_eq(self): return np.zeros(0)


	def _isinside(self, X, tol = TOL):
		return self._isinside_quad(X, tol = tol) 

	def _pdf(self, X):
		# Mahalanobis distance
		d2 = np.sum(self.Linv.dot(X.T - self.mean.reshape(-1,1))**2, axis = 0)
		# Normalization term
		p = np.exp(-0.5*d2) / np.sqrt((2*np.pi)**len(self) * np.abs(scipy.linalg.det(self.cov)))
		if self.truncate is not None:
			# Normalization term
			p /= (1-self.truncate)
			# probability is zero for those points outside of the domain
			p *= self.isinside(X)
			pass
		return p
