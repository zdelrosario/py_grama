from __future__ import division

import numpy as np
import cvxpy as cp

import scipy.linalg
from scipy.linalg import orth
from scipy.optimize import nnls
from scipy.spatial import ConvexHull

from functools import lru_cache

try:
	from functools import cached_property
except ImportError:
	from backports.cached_property import cached_property

from .domain import TOL, DEFAULT_CVXPY_KWARGS
from .linquad import LinQuadDomain
from .box import BoxDomain
from ..misc import merge
from ..geometry import unique_points
from .euclidean import TOL




def _hull_to_linineq(X):
	r"""
	"""
	if X.shape[0] == 1:
		A = None
		b = None
		vertices = X
		A_eq = np.eye(X.shape[1])
		b_eq = X.flatten()
		return A, b, A_eq, b_eq, vertices

	# Find a low-dimensional subspace on which this data lives
	Xc = np.mean(X, axis = 0)
	Xdiff = (X.T - Xc.reshape(-1,1)).T
	U, s, VT = scipy.linalg.svd(Xdiff, full_matrices = False)
	I = np.isclose(s, 0)
	Q, _ = scipy.linalg.qr(VT[~I].T, mode = 'full')
	dim = np.sum(~I)
	# component where coordinates change
	Qpar = Q[:,0:dim]
	# orthogonal complement
	Qperp = Q[:, dim:]

	Y = X @ Qpar
	if Y.shape[1] > 1:
		hull = ConvexHull(Y)
		A = hull.equations[:,:-1] @ Qpar.T
		b = -hull.equations[:,-1]
		vertices = X[hull.vertices]
	else:
		# this is an interval which is not handled by Qhull
		A = np.array([1, -1]).reshape(2,1) @ Qpar.T
		b = np.array([np.max(Y), -np.min(Y)])
		vertices = np.array([X[np.argmax(Y)], X[np.argmin(Y)]])
	
	A_eq = Qperp.T
	b_eq = np.mean( X @ Qperp, axis = 0)
	return A, b, A_eq, b_eq, vertices


class _Extent:
	def __init__(self, domain):

		self.domain = domain
		self.alpha = cp.Variable(len(domain.X), nonneg = True) # Convex combination parameters
		self.beta = cp.Variable()           # step length
		self.x_norm = cp.Parameter(len(domain)) # starting point in the domain
		self.p_norm = cp.Parameter(len(domain)) # direction in the domain
	
		self.obj = cp.Maximize(self.beta)
		self.constraints = [
			self.domain._X_norm.T @ self.alpha == self.p_norm * self.beta + self.x_norm,
			cp.sum(self.alpha) == 1
		] 
		self.constraints += LinQuadDomain._build_constraints_norm(domain, self.x_norm) 
		
		self.prob = cp.Problem(self.obj, self.constraints)

	def __call__(self, x, p, **kwargs):
		kwargs['warm_start'] = False
		self.x_norm.value = self.domain.normalize(x)
		self.p_norm.value = self.domain.normalize(x+p)-self.domain.normalize(x)

		#self.prob.solve(verbose = True)
		self.prob.solve(**merge(self.domain.kwargs, kwargs))
		try:
			return float(self.beta.value)
		except Exception as e:
			print(e)
			return 0

class ConvexHullDomain(LinQuadDomain):
	r"""Define a domain that is the interior of a convex hull of points.

	Given a set of points :math:`\lbrace x_i \rbrace_{i=1}^M\subset \mathbb{R}^m`,
	construct a domain from their convex hull:

	.. math::
	
		\mathcal{D} := \left\lbrace \sum_{i=1}^M \alpha_i x_i : \sum_{i=1}^M \alpha_i = 1, \ \alpha_i \ge 0 \right\rbrace \subset \mathbb{R}^m.

	In additionally any linear equality, linear inequality, and quadratic inequality constraints can be included.
	

	Parameters
	----------
	X: array-like (M, m)
		Points from which to build the convex hull of points.
	A: array-like (m,n)
		Matrix in left-hand side of inequality constraint
	b: array-like (m,)
		Vector in right-hand side of the ineqaluty constraint
	A_eq: array-like (p,n)
		Matrix in left-hand side of equality constraint
	b_eq: array-like (p,) 
		Vector in right-hand side of equality constraint
	lb: array-like (n,)
		Vector of lower bounds 
	ub: array-like (n,)
		Vector of upper bounds 
	Ls: list of array-likes (p,m)
		List of matrices with m columns defining the quadratic constraints
	ys: list of array-likes (m,)
		Centers of the quadratic constraints
	rhos: list of positive floats 
		Radii of quadratic constraints
	names: list of strings, optional
		Names for each of the parameters in the space
	kwargs: dict, optional
		Additional parameters to be passed to cvxpy Problem.solve() 
	"""

	def __init__(self, X, A = None, b = None, lb = None, ub = None, 
		A_eq = None, b_eq = None, Ls = None, ys = None, rhos = None,
		names = None, tol = TOL, **kwargs):

		X = np.atleast_2d(X)
		I = unique_points(X)
		self._X = np.copy(X[I])
		self._init_names(names)
		self.tol = tol

		# Start setting default values
		self._lb = self._init_lb(lb)
		self._ub = self._init_ub(ub)
		self._A, self._b = self._init_ineq(A, b)
		self._A_eq, self._b_eq = self._init_eq(A_eq, b_eq)	
		self._Ls, self._ys, self._rhos = self._init_quad(Ls, ys, rhos)
		
		# TODO: should we consider reducing dimension via rotation
		# if the points are colinear ?

		# Setup the lower and upper bounds to improve conditioning
		# when solving LPs associated with domain features
		self._norm_lb = np.min(self._X, axis = 0)
		self._norm_ub = np.max(self._X, axis = 0)
		self._X_norm = self.normalize(self.X)
		
		self.kwargs = merge(DEFAULT_CVXPY_KWARGS, kwargs)

		self._extent = _Extent(self) 


	def _is_box_domain(self):
		return False

	def __str__(self):
		ret = "<ConvexHullDomain on R^%d based on %d points" % (len(self), len(self._X_norm))
		if len(self._Ls) > 0:
			ret += "; %d quadratic constraints" % (len(self._Ls),)
		if self._A.shape[0] > 0:
			ret += "; %d linear inequality constraints" % (self._A.shape[0], )
		if self._A_eq.shape[0] > 0:
			ret += "; %d linear equality constraints" % (self._A_eq.shape[0], )
		
		ret += ">"
		return ret

	def chebyshev_center(self):
		raise NotImplementedError

	@lru_cache(maxsize = None)
	def to_linineq(self, **kwargs):
		r""" Convert the domain into a LinIneqDomain

		"""
		A, b, A_eq, b_eq, vertices = _hull_to_linineq(self._X)
		dom_hull = LinQuadDomain(A = A, b = b, A_eq = A_eq, b_eq = b_eq, names = self.names, **kwargs)
		dom_hull.vertices = np.copy(vertices)
		# Add back in non-hull constraints
		dom = dom_hull.add_constraints(A = self.A, b = self.b, A_eq = self.A_eq, b_eq = self.b_eq,
				Ls = self.Ls, ys = self.ys, rhos = self.rhos)
		return dom


	def coefficients(self, x, **kwargs):
		r""" Find the coefficients of the convex combination of elements in the space yielding x

		"""
		x_norm = self.normalize(x)		
		A = np.vstack([self._X_norm.T, np.ones( (1,len(self._X_norm)) )])
		b = np.hstack([x_norm, 1])
		alpha, rnorm = nnls(A, b)
		return alpha

	@property
	def X(self):
		return np.copy(self._X)

	def __len__(self):
		return self._X.shape[1]


	def _sample(self, draw = 1):
		if len(self.X) == 1:
			return np.outer(np.ones(draw), self.X)

		try:
			# Try a quicker method to sample from the domain if there are only two points
			assert len(self._X) == 2
			alphas = np.random.uniform(0,1, size = draw)
			X = np.vstack([self._X[0] * alpha + self._X[1]*(1-alpha) for alpha in alphas])
			# These points automatically satisfy the convex combination constraint
			# We then check if the remaining constraints are satisfied
			# if not, we error out and revert to hit and run sampling
			assert np.all(self._isinside_bounds(X))
			assert np.all(self._isinside_ineq(X))
			assert np.all(self._isinside_eq(X))
			assert np.all(self._isinside_quad(X))
			return X
		except AssertionError:
			pass

		if len(self) <= 3:
			dom = self.to_linineq()
			return dom.sample(draw) 
		else:
				
			return super(ConvexHullDomain, self)._sample(draw = draw)

	def _build_constraints(self, x):
		
		alpha = cp.Variable(len(self.X), name = 'alpha')
		constraints = [x == alpha @ self._X.T,  alpha >=0, cp.sum(alpha) == 1]
		constraints += LinQuadDomain._build_constraints(self, x)
		return constraints
		
	def _build_constraints_norm(self, x_norm):
		alpha = cp.Variable(len(self._X_norm), name = 'alpha')
		constraints = [x_norm == alpha @ self._X_norm, alpha >=0, cp.sum(alpha) == 1]
		constraints += LinQuadDomain._build_constraints_norm(self, x_norm)
		return constraints
	
	def _isinside(self, X, tol = TOL):

		# Check that the points are in the convex hull
		inside = np.zeros(X.shape[0], dtype = np.bool)
		
		for i, xi in enumerate(X):
			alpha = self.coefficients(xi)
			rnorm = np.linalg.norm( xi - self._X.T.dot(alpha))
			inside[i] = (rnorm < tol)

		# Now check linear inequality/equality constraints and quadratic constraints
		inside &= self._isinside_bounds(X, tol = tol)
		inside &= self._isinside_ineq(X, tol = tol)	
		inside &= self._isinside_eq(X, tol = tol)
		inside &= self._isinside_quad(X, tol = tol)

		return inside

	def add_constraints(self, A = None, b = None, lb = None, ub = None, A_eq = None, b_eq = None,
		Ls = None, ys = None, rhos = None):
		lb = self._init_lb(lb)
		ub = self._init_ub(ub)
		A, b = self._init_ineq(A, b)
		A_eq, b_eq = self._init_eq(A_eq, b_eq)
		Ls, ys, rhos = self._init_quad(Ls, ys, rhos)
		
		A = np.vstack([self.A, A])
		b = np.hstack([self.b, b])
		
		lb = np.maximum(lb, self.lb)
		ub = np.minimum(ub, self.ub)

		A_eq = np.vstack([self.A_eq, A_eq])
		b_eq = np.hstack([self.b_eq, b_eq])
		
		Ls = self.Ls + Ls
		ys = self.ys + ys
		rhos = self.rhos + rhos

		return ConvexHullDomain(self.X, A = A, b = b, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq,
			Ls = Ls, ys = ys, rhos = rhos, names = self.names, tol = self.tol, **self.kwargs)  



	@cached_property
	def _A_eq_basis(self):
		try: 
			if len(self.A_eq) == 0: raise AttributeError
			Qeq = orth(self.A_eq.T)
		except AttributeError:
			Qeq = np.zeros((len(self),0))
		
		# Check if points are colinear; if so add the corresponding equality constraint	
		Xc = np.mean(self.X, axis = 0)
		Xdiff = (self.X.T - Xc.reshape(-1,1)).T
		U, s, VT = scipy.linalg.svd(Xdiff, full_matrices = False)
		
		I = np.isclose(s, 0)
		if np.sum(I) > 0:
			# Find an orthogonal basis for the nullspace of the nonzero right singular vectors
			# (we do this to avoid computing a full singular value decomposition as there may be many 
			# points defining the convex hull domain)
			Q, _ = scipy.linalg.qr(VT[~I].T, mode = 'full')
			Q = Q[:, np.sum(~I):]
			Qeq = orth(np.hstack([Qeq, Q]))
		return Qeq	
