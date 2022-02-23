r""" LinQuadDomain definition

Philosophically, LinQuad domains consist of those convex domains specified by a combination
of linear inequality, linear equality, and quadratic inequality constraints.

From a code perspective, this is where a dependency on CVXPY is introduced to
handle domain properities from within this domain

"""

from __future__ import division

import numpy as np
from .domain import TOL#, DEFAULT_CVXPY_KWARGS
from .euclidean import EuclideanDomain
from .tensor import TensorProductDomain

from ..misc import merge

class LinQuadDomain(EuclideanDomain):
	r"""A domain specified by a combination of linear (in)equality constraints and convex quadratic constraints


	Here we define a domain that is specified in terms of bound constraints,
	linear inequality constraints, linear equality constraints, and quadratic constraints.

	.. math::

		\mathcal{D} := \left \lbrace
			\mathbf{x} : \text{lb} \le \mathbf{x} \le \text{ub}, \
			\mathbf{A} \mathbf{x} \le \mathbf{b}, \
			\mathbf{A}_{\text{eq}} \mathbf{x} = \mathbf{b}_{\text{eq}}, \
			\| \mathbf{L}_i (\mathbf{x} - \mathbf{y}_i)\|_2 \le \rho_i
		\right\rbrace \subset \mathbb{R}^m


	Parameters
	----------
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
	def __init__(self, A = None, b = None,
		lb = None, ub = None,
		A_eq = None, b_eq = None,
		Ls = None, ys = None, rhos = None,
		names = None, **kwargs):

		self.tol = 1e-6
		# Determine dimension of space
		self._init_dim(lb = lb, ub = ub, A = A, A_eq = A_eq, Ls = Ls)

		# Start setting default values
		self._lb = self._init_lb(lb)
		self._ub = self._init_ub(ub)
		self._A, self._b = self._init_ineq(A, b)
		self._A_eq, self._b_eq = self._init_eq(A_eq, b_eq)
		self._Ls, self._ys, self._rhos = self._init_quad(Ls, ys, rhos)

		self._init_names(names)


		# self.kwargs = merge(DEFAULT_CVXPY_KWARGS, kwargs)

	def __str__(self):
		ret = "<%s on R^%d" % (self.__class__.__name__, len(self))
		if len(self._Ls) > 0:
			ret += "; %d quadratic constraints" % (len(self._Ls),)
		if self._A.shape[0] > 0:
			ret += "; %d linear inequality constraints" % (self._A.shape[0], )
		if self._A_eq.shape[0] > 0:
			ret += "; %d linear equality constraints" % (self._A_eq.shape[0], )
		ret +=">"

		return ret


	################################################################################
	# Initialization helpers
	################################################################################
	def _init_dim(self, lb = None, ub = None, A = None, A_eq = None, Ls = None):
		"""determine the dimension of the space we are working on"""
		if lb is not None:
			if isinstance(lb, (int, float)):
				m = 1
			else:
				m = len(lb)
		elif ub is not None:
			if isinstance(ub, (int, float)):
				m = 1
			else:
				m = len(ub)
		elif A is not None:
			m = len(A[0])
		elif A_eq is not None:
			m = len(A_eq[0])
		elif Ls is not None:
			m = len(Ls[0][0])
		else:
			raise Exception("Could not determine dimension of space")

		self._dimension = m


	def _init_lb(self, lb):
		if lb is None:
			return -np.inf*np.ones(len(self))
		else:
			if isinstance(lb, (int, float)):
				lb = [lb]
			assert len(lb) == len(self), "Lower bound has wrong dimensions"
			return np.array(lb)

	def _init_ub(self, ub):
		if ub is None:
			return np.inf*np.ones(len(self))
		else:
			if isinstance(ub, (int, float)):
				ub = [ub]
			assert len(ub) == len(self), "Upper bound has wrong dimensions"
			return np.array(ub)

	def _init_ineq(self, A, b):
		if A is None and b is None:
			A = np.zeros((0,len(self)))
			b = np.zeros((0,))
		elif A is not None and b is not None:
			A = np.array(A)
			b = np.array(b)
			if len(b.shape) == 0:
				b = b.reshape(1)

			assert len(b.shape) == 1, "b must have only one dimension"

			if len(A.shape) == 1 and len(b) == 1:
				A = A.reshape(1,-1)

			assert A.shape[1] == len(self), "A has wrong number of columns"
			assert A.shape[0] == b.shape[0], "The number of rows of A and b do not match"
		else:
			raise AssertionError("If using inequality constraints, both A and b must be specified")
		return A, b

	def _init_eq(self, A_eq, b_eq):
		if A_eq is None and b_eq is None:
			A_eq = np.zeros((0,len(self)))
			b_eq = np.zeros((0,))
		elif A_eq is not None and b_eq is not None:
			A_eq = np.array(A_eq)
			b_eq = np.array(b_eq)
			if len(b_eq.shape) == 0:
				b_eq = b_eq.reshape(1)

			assert len(b_eq.shape) == 1, "b_eq must have only one dimension"

			if len(A_eq.shape) == 1 and len(b_eq) == 1:
				A_eq = A_eq.reshape(1,-1)

			assert A_eq.shape[1] == len(self), "A_eq has wrong number of columns"
			assert A_eq.shape[0] == b_eq.shape[0], "The number of rows of A_eq and b_eq do not match"
		else:
			raise AssertionError("If using equality constraints, both A_eq and b_eq must be specified")

		return A_eq, b_eq

	def _init_quad(self, Ls, ys, rhos):
		if Ls is None and ys is None and rhos is None:
			_Ls = []
			_ys = []
			_rhos = []
		elif Ls is not None and ys is not None and rhos is not None:
			assert len(Ls) == len(ys) == len(rhos), "Length of all quadratic constraints must be the same"

			_Ls = []
			_ys = []
			_rhos = []
			for L, y, rho in zip(Ls, ys, rhos):
				assert len(L[0]) == len(self), "dimension of L doesn't match the domain"
				assert len(y) == len(self), "Dimension of center doesn't match the domain"
				assert rho > 0, "Radius must be positive"
				_Ls.append(np.array(L))
				_ys.append(np.array(y))
				_rhos.append(rho)
				# TODO: If constraint is rank-1, should we implicitly convert to a linear inequality constriant
		else:
			raise AssertionError("If providing quadratic constraint, each of Ls, ys, and rhos must be defined")
		return _Ls, _ys, _rhos

	################################################################################
	# Simple properties
	################################################################################
	def __len__(self): return self._dimension

	@property
	def lb(self): return self._lb

	@property
	def ub(self): return self._ub

	@property
	def A(self): return self._A

	@property
	def b(self): return self._b

	@property
	def A_eq(self): return self._A_eq

	@property
	def b_eq(self): return self._b_eq

	@property
	def Ls(self): return self._Ls

	@property
	def ys(self): return self._ys

	@property
	def rhos(self): return self._rhos




	################################################################################
	# Normalization
	################################################################################
	def _normalized_domain(self, **kwargs):
		names_norm = [name + ' (normalized)' for name in self.names]

		return LinQuadDomain(lb = self.lb_norm, ub = self.ub_norm, A = self.A_norm, b = self.b_norm,
			A_eq = self.A_eq_norm, b_eq = self.b_eq_norm, Ls = self.Ls_norm, ys = self.ys_norm, rhos = self.rhos_norm,
			names = names_norm, **merge(self.kwargs, kwargs))

	def _isinside(self, X, tol = TOL):
		return self._isinside_bounds(X, tol = tol) & self._isinside_ineq(X, tol = tol) & self._isinside_eq(X, tol = tol) & self._isinside_quad(X, tol = tol)

	def _extent(self, x, p):
		# Check that direction satisfies equality constraints to a tolerance
		if self.A_eq.shape[0] == 0 or np.all(np.abs(self.A_eq.dot(p) ) < self.tol):
			return min(self._extent_bounds(x, p), self._extent_ineq(x, p), self._extent_quad(x, p))
		else:
			return 0.

	################################################################################
	# Convex Solver Functions
	################################################################################

	# def _build_constraints_norm(self, x_norm):
	# 	r""" Build the constraints corresponding to the domain given a vector x
	# 	"""
	# 	constraints = []

	# 	# Numerical issues emerge with unbounded constraints
	# 	I = np.isfinite(self.lb_norm)
	# 	if np.sum(I) > 0:
	# 		constraints.append( self.lb_norm[I] <= x_norm[I])

	# 	I = np.isfinite(self.ub_norm)
	# 	if np.sum(I) > 0:
	# 		constraints.append( x_norm[I] <= self.ub_norm[I])

	# 	if self.A.shape[0] > 0:
	# 		constraints.append( x_norm.__rmatmul__(self.A_norm) <= self.b_norm)
	# 	if self.A_eq.shape[0] > 0:
	# 		constraints.append( x_norm.__rmatmul__(self.A_eq_norm) == self.b_eq_norm)

	# 	for L, y, rho in zip(self.Ls_norm, self.ys_norm, self.rhos_norm):
	# 		if len(L) > 1:
	# 			constraints.append( cp.norm( L @ x_norm - L @ y) <= rho )
	# 		elif len(L) == 1:
	# 			constraints.append( cp.norm(L @ x_norm - L @ y) <= rho)

	# 	return constraints


	# def _build_constraints(self, x):
	# 	r""" Build the constraints corresponding to the domain given a vector x
	# 	"""
	# 	constraints = []

	# 	# Numerical issues emerge with unbounded constraints
	# 	I = np.isfinite(self.lb)
	# 	if np.sum(I) > 0:
	# 		constraints.append( self.lb[I] <= x[I])

	# 	I = np.isfinite(self.ub)
	# 	if np.sum(I) > 0:
	# 		constraints.append( x[I] <= self.ub[I])

	# 	if self.A.shape[0] > 0:
	# 		constraints.append( x.__rmatmul__(self.A) <= self.b)
	# 	if self.A_eq.shape[0] > 0:
	# 		constraints.append( x.__rmatmul__(self.A_eq) == self.b_eq)

	# 	for L, y, rho in zip(self.Ls, self.ys, self.rhos):
	# 		if len(L) > 1:
	# 			constraints.append( cp.norm(L @ x - L.dot(y)) <= rho )
	# 		elif len(L) == 1:
	# 			constraints.append( cp.norm(L @ x - L.dot(y)) <= rho)

	# 	return constraints


	################################################################################
	#
	################################################################################

	def add_constraints(self, A = None, b = None, lb = None, ub = None, A_eq = None, b_eq = None,
		Ls = None, ys = None, rhos = None):
		r"""Add new constraints to the domain
		"""
		lb = self._init_lb(lb)
		ub = self._init_ub(ub)
		A, b = self._init_ineq(A, b)
		A_eq, b_eq = self._init_eq(A_eq, b_eq)
		Ls, ys, rhos = self._init_quad(Ls, ys, rhos)

		# Update constraints
		lb = np.maximum(lb, self.lb)
		ub = np.minimum(ub, self.ub)

		A = np.vstack([self.A, A])
		b = np.hstack([self.b, b])

		A_eq = np.vstack([self.A_eq, A_eq])
		b_eq = np.hstack([self.b_eq, b_eq])

		Ls = self.Ls + Ls
		ys = self.ys + ys
		rhos = self.rhos + rhos

		if len(Ls) > 0:
			return LinQuadDomain(lb = lb, ub = ub, A = A, b = b, A_eq = A_eq, b_eq = b_eq,
				 Ls = Ls, ys = ys, rhos = rhos, names = self.names)
		elif len(b) > 0 or len(b_eq) > 0:
			from .linineq import LinIneqDomain
			return LinIneqDomain(lb = lb, ub = ub, A = A, b = b, A_eq = A_eq, b_eq = b_eq, names = self.names)
		else:
			from .box import BoxDomain
			return BoxDomain(lb = lb, ub = ub, names = self.names)

	def __and__(self, other):
		if isinstance(other, LinQuadDomain) or (isinstance(other, TensorProductDomain) and other.is_linquad_domain):
			return self.add_constraints(lb = other.lb, ub = other.ub,
				A = other.A, b = other.b, A_eq = other.A_eq, b_eq = other.b_eq,
				Ls = other.Ls, ys = other.ys, rhos = other.rhos)
		else:
			raise NotImplementedError

	def __rand__(self, other):
		return self.__and__(other)
