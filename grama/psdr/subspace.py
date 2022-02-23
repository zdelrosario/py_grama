# Subspace based dimension reduction techniques
from __future__ import division, print_function

import os
# Necessary for running in headless enviornoments
if 'DISPLAY' not in os.environ:
	import matplotlib
	matplotlib.use("Agg")
import matplotlib.pyplot as plt


import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse


import cvxpy as cp
import cvxopt

from .pgf import PGF
from .domains.domain import DEFAULT_CVXPY_KWARGS
from .misc import merge

__all__ = ['SubspaceBasedDimensionReduction',
	'ActiveSubspace', 
	]

class SubspaceBasedDimensionReduction(object):
	r""" Abstract base class for Subspace-Based Dimension Reduction

	Given a function :math:`f : \mathcal{D} \to \mathbb{R}`, 
	subspace-based dimension reduction identifies a subspace, 
	described by a matrix :math:`\mathbf{U} \in \mathbb{R}^{m\times n}`
	with orthonormal columns for some :math:`n \le m`.

	"""
	@property
	def U(self):
		""" A matrix defining the 'important' directions

		Returns
		-------
		np.ndarray (m, n):
			Matrix with orthonormal columns defining the important directions in decreasing order
			of precidence.
		"""
		raise NotImplementedError


	def shadow_plot(self, X = None, fX = None, dim = 1, U = None, ax = 'auto', pgfname = None):
		r""" Draw a shadow plot


		Parameters
		----------
		X: array-like (N,m)
			Input coordinates for function samples
		fX: array-like (N,)
			Values of function at sample points
		dim: int, [1,2]
			Dimension of shadow plot
		U: array-like (?,m); optional
			Subspace onto which to project the data; defaults to the subspace identifed by this class
		ax: 'auto', matplotlib.pyplot.axis, or None
			Axis on which to draw the shadow plot

		Returns
		-------
		ax: matplotlib.pyplot.axis
			Axis on which the plot is drawn
		"""
		if ax == 'auto':
			if dim == 1:
				fig, ax = plt.subplots(figsize = (6,6))
			else:
				# Hack so that plot is approximately square after adding colorbar 
				fig, ax = plt.subplots(figsize = (7.5,6))
	
		if X is None:
			X = self.X
	
		# Check dimensions
		X = np.atleast_2d(X)
		assert X.shape[1] == len(self), "Samples do not match dimension of space"	

		if U is None:
			U = self.U
		else:
			if len(U.shape) == 1:
				U = U.reshape(len(self),1)
			else:
				assert U.shape[0] == len(self), "Dimensions do not match"

		
		if dim == 1:
			if ax is not None:
				ax.plot(X.dot(U[:,0]), fX, 'k.')
				ax.set_xlabel(r'active coordinate $\mathbf{u}^\top \mathbf{x}$')
				ax.set_ylabel(r'$f(\mathbf{x})$')

			if pgfname is not None:
				pgf = PGF()
				pgf.add('y', X.dot(U[:,0]))
				pgf.add('fX', fX)
				pgf.write(pgfname)

		elif dim == 2:
			Y = U[:,0:2].T.dot(X.T).T
			
			if ax is not None:
				sc = ax.scatter(Y[:,0], Y[:,1], c = fX.flatten(), s = 3)
				ax.set_xlabel(r'active coordinate 1 $\mathbf{u}_1^\top \mathbf{x}$')
				ax.set_ylabel(r'active coordinate 2 $\mathbf{u}_2^\top \mathbf{x}$')

				plt.colorbar(sc).set_label('f(x)')
			
			if pgfname is not None:
				pgf = PGF()
				pgf.add('y1', Y[:,0])
				pgf.add('y2', Y[:,1])
				pgf.add('fX', fX.flatten())
				pgf.write(pgfname)

		else:
			raise NotImplementedError		

		return ax

	def shadow_envelope(self, X, fX, ax = None, ngrid = None, pgfname = None, verbose = True, U = None, **kwargs):
		r""" Draw a 1-d shadow plot of a large number of function samples

		Returns
		-------
		y: np.ndarray
			Projected coordinates
		lb: np.ndarray
			piecewise linear lower bound values
		ub: np.ndarray
			piecewise linear upper bound values
		"""
		if U is None:
			U = self.U[:,0]		
		else:
			if len(U.shape) > 1:
				U = U[:,0]

		# Since this is for plotting purposes, we reduce accuracy to 3 digits	
		solver_kwargs = {'verbose': verbose, 'solver': 'OSQP', 'eps_abs': 1e-3, 'eps_rel': 1e-3}				

		X = np.array(X)
		fX = np.array(fX)
		assert len(X) == len(fX), "Number of inputs did not match number of outputs"
		if len(fX.shape) > 1:
			fX = fX.flatten()
			assert len(fX) == len(X), "Expected fX to be a vector"

		y = X.dot(U)
		if ngrid is None:
			# Determine the minimum number of bins
			ngrid = 25
			while True:
				yy = np.linspace(np.min(y), np.max(y), ngrid)
				h = yy[1] - yy[0]	
				if ngrid == 3:
					break 
				# Make sure we have at least two entries in every bin:
				items, counts = np.unique(np.floor( (y - yy[0])/h), return_counts = True)
				# We ignore the last count of the bins as that is the right endpoint and will only ever have one
				if (np.min(counts[:-1]) >= 5) and len(items) == ngrid:
					break
				else:
					ngrid -= 1
		else:
			yy = np.linspace(np.min(y), np.max(y), ngrid)
			h = yy[1] - yy[0]

		h = float(h)

		# Build the piecewise linear interpolation matrix
		j = np.floor( (y - yy[0])/h ).astype(np.int)
		row = []
		col = []
		val = []

		# Points not at the right endpoint
		row += np.arange(len(y)).tolist()
		col += j.tolist()
		val += ((  (yy[0]+ (j+1)*h) - y )/h).tolist()

		# Points not at the right endpoint
		I = (j != len(yy) - 1)
		row += np.argwhere(I).flatten().tolist()
		col += (j[I]+1).tolist()
		val += ( (y[I] - (yy[0] + j[I]*h)  )/h).tolist()

		A = scipy.sparse.coo_matrix((val, (row, col)), shape = (len(y), len(yy)))
		A = cp.Constant(A)
		ub = cp.Variable(len(yy))
		#ub0 = [ max(max(fX[j == i]), max(fX[j== i+1]))  for i in np.arange(0,ngrid-1)] +[max(fX[j == ngrid - 1])]
		#ub.value = np.array(ub0).flatten()
		prob = cp.Problem(cp.Minimize(cp.sum(ub)), [A*ub >= fX.flatten()])
		prob.solve(**solver_kwargs)
		ub = ub.value
		
		lb = cp.Variable(len(yy))
		#lb0 = [ min(min(fX[j == i]), min(fX[j== i+1]))  for i in np.arange(0,ngrid-1)] +[min(fX[j == ngrid - 1])]
		#lb.value = np.array(lb0).flatten()
		prob = cp.Problem(cp.Maximize(cp.sum(lb)), [A*lb <= fX.flatten()])
		prob.solve(**solver_kwargs)
		lb = lb.value

		if ax is not None:
			ax.fill_between(yy, lb, ub, **kwargs) 

		if pgfname is not None:
			pgf = PGF()
			pgf.add('y', yy)
			pgf.add('lb', lb)
			pgf.add('ub', ub)	
			pgf.write(pgfname)
		
		return y, lb, ub


	def _init_dim(self, X = None, grads = None):
		if X is not None and len(X) > 0:
			self._dimension = len(X[0])
		elif grads is not None:
			self._dimension = len(grads[0])
		else:
			raise Exception("Could not determine dimension of ambient space")


	def __len__(self):
		return self._dimension

	@property
	def X(self):
		return np.zeros((0,len(self)))
	
	@property
	def fX(self):
		return np.zeros((0,len(self)))

	@property
	def grads(self):
		return np.zeros((0,len(self)))

	def _fix_subspace_signs(self, U, X = None, fX = None, grads = None):
		r""" Orient the subspace so that the average slope is positive

		Since subspaces have no associated direction (they are invariant to a sign flip)
		here we fix the sign such that the function is increasing on average along the direction
		u_i.  This approach uses either gradient or sample information, with a preference for
		gradient information if it is availible.
		"""
		if grads is not None and len(grads) > 0:
			return self._fix_subspace_signs_grads(U, grads)
		else:
			return self._fix_subspace_signs_samps(U, X, fX)	

	def _fix_subspace_signs_samps(self, U, X, fX):
		sgn = np.zeros(len(U[0]))
		for k in range(len(U[0])):
			for i in range(len(X)):
				for j in range(i+1, len(X)):
					denom = U[:,k] @ (X[i] - X[j])
					if np.abs(denom) > 0:
						sgn[k] += (fX[i] - fX[j])/denom

		# If the sign is zero, keep the current orientation
		sgn[sgn == 0] = 1
		return U.dot(np.diag(np.sign(sgn)))	

	def _fix_subspace_signs_grads(self, U, grads):
		return U.dot(np.diag(np.sign(np.mean(grads.dot(U), axis = 0))))

	
	def approximate_lipschitz(self, X = None, fX = None, grads = None,  dim = None):
		r""" Approximate the Lipschitz matrix on the low-dimensional subspace
		"""
		raise NotImplementedError


class ActiveSubspace(SubspaceBasedDimensionReduction):
	r"""Computes the active subspace gradient samples

	Given the function :math:`f:\mathcal{D} \to \mathbb{R}`,
	the active subspace is defined as the eigenvectors corresponding to the 
	largest eigenvalues of the average outer-product of gradients:

	.. math::

		\mathbf{C} := \int_{\mathbf{x}\in \mathcal{D}} \nabla f(\mathbf{x}) \nabla f(\mathbf{x})^\top \  \mathrm{d}\mathbf{x}
		\in \mathbb{R}^{m\times m}.

	By default, this class assumes that we are provided with gradients
	evaluated at random samples over the domain and estimates the matrix :math:`\mathbf{C}`
	using Monte-Carlo integration. However, if provided a weight corresponding to a quadrature rule,
	this will be used instead to approximate this matrix; i.e.,
		
	.. math::

		\mathbf{C} \approx \sum_{i=1}^N w_i \nabla f(\mathbf{x}_i) \nabla f(\mathbf{x}_i)^\top.

	"""
	def __init__(self):
		self._U = None
		self._s = None


	def fit(self, grads, weights = None):
		r""" Find the active subspace

		Parameters
		----------
		grads: array-like (N,m)
			Gradient samples of function (tacitly assumed to be uniform on the domain
			or from a quadrature rule with corresponding weight).
		weights: array-like (N,), optional
			Weights corresponding to a quadrature rule associated with the samples of the gradient.

		"""
		self._init_dim(grads = grads)

		self._grads = np.array(grads).reshape(-1,len(self))
		N = len(self._grads)
		if weights is None:
			weights = np.ones(N)/N
			
		self._weights = np.array(weights)
		self._U, self._s, VT = scipy.linalg.svd(np.sqrt(self._weights)*self._grads.T)
		# Pad s with zeros if we don't have as many gradient samples as dimension of the space
		self._s = np.hstack([self._s, np.zeros(self._dimension - len(self._s))])
		self._C = self._U @ np.diag(self._s**2) @ self._U.T

		# Fix +/- scaling so average gradient is positive	
		self._U = self._fix_subspace_signs_grads(self._U, self._grads)		


	def fit_function(self, fun, N_gradients):
		r""" Automatically estimate active subspace using a quadrature rule

		Parameters
		----------
		fun: Function
			function object for which to estimate the active subspace via the average outer-product of gradients
		N_gradients: int
			Maximum number of gradient samples to use 
		"""
		X, w = fun.domain.quadrature_rule(N_gradients)
		grads = fun.grad(X)
		self.fit(grads, w)
			

	@property
	def U(self):
		return np.copy(self._U)

	@property
	def C(self):
		return self._C

	@property
	def singvals(self):
		return self._s

	# TODO: Plot of eigenvalues (with optional boostrapped estimate)

	# TODO: Plot of eigenvector angles with bootstrapped replicates.


