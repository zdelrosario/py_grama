from __future__ import division

import numpy as np

# import cvxpy as cp

from .domain import TOL
from .linquad import LinQuadDomain

from ..misc import merge

class LinIneqDomain(LinQuadDomain):
	r"""A domain specified by a combination of linear equality and inequality constraints.

	Here we build a domain specified by three kinds of constraints:
	bound constraints :math:`\text{lb} \le \mathbf{x} \le \text{ub}`,
	inequality constraints :math:`\mathbf{A} \mathbf{x} \le \mathbf{b}`,
	and equality constraints :math:`\mathbf{A}_{\text{eq}} \mathbf{x} = \mathbf{b}_{\text{eq}}`:

	.. math::

		\mathcal{D} := \left \lbrace
			\mathbf{x} : \text{lb} \le \mathbf{x} \le \text{ub}, \
			\mathbf{A} \mathbf{x} \le \mathbf{b}, \
			\mathbf{A}_{\text{eq}} \mathbf{x} = \mathbf{b}_{\text{eq}}
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
	kwargs: dict, optional
		Additional parameters to pass to solvers

	"""
	def __init__(self, A = None, b = None, lb = None, ub = None, A_eq = None, b_eq = None, names = None, **kwargs):
		LinQuadDomain.__init__(self, A = A, b = b, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq, names = names, **kwargs)

	def _isinside(self, X, tol = TOL):
		return self._isinside_bounds(X, tol = tol) & self._isinside_ineq(X, tol = tol) & self._isinside_eq(X, tol = tol)

	def _extent(self, x, p):
		return min(self._extent_bounds(x, p), self._extent_ineq(x, p))

	def _normalized_domain(self, **kwargs):
		names_norm = [name + ' (normalized)' for name in self.names]
		return LinIneqDomain(lb = self.lb_norm, ub = self.ub_norm, A = self.A_norm, b = self.b_norm,
			A_eq = self.A_eq_norm, b_eq = self.b_eq_norm, names = names_norm, **merge(self.kwargs, kwargs))


	# def chebyshev_center(self):
	# 	r"""Estimates the Chebyshev center using the constrainted least squares approach

	# 	Solves the linear program finding the radius :math:`r` and Chebyshev center :math:`\mathbf{x}`.

	# 	.. math::

	# 		\max_{r\in \mathbb{R}^+, \mathbf{x} \in \mathcal{D}} &\  r \\
	# 		\text{such that} & \ \mathbf{a}_i^\top \mathbf{x} + r \|\mathbf{a}_i\|_2 \le b_i

	# 	where we have expressed the domain in terms of the linear inequality constraints
	# 	:math:`\mathcal{D}=\lbrace \mathbf{x} : \mathbf{A}\mathbf{x} \le \mathbf{b}\rbrace`
	# 	and :math:`\mathbf{a}_i^\top` are the rows of :math:`\mathbf{A}` as described in [BVNotes]_.


	# 	Returns
	# 	-------
	# 	center: np.ndarray(m,)
	# 		Center of the domain
	# 	radius: float
	# 		radius of largest circle inside the domain

	# 	References
	# 	----------
	# 	.. [BVNotes] https://see.stanford.edu/materials/lsocoee364a/04ConvexOptimizationProblems.pdf, page 4-19.
	# 	"""
	# 	m, n = self.A.shape

	# 	# Merge the bound constraints into A
	# 	A = self.A_aug
	# 	b = self.b_aug

	# 	# See p.4-19 https://see.stanford.edu/materials/lsocoee364a/04ConvexOptimizationProblems.pdf
	# 	#
	# 	normA = np.sqrt( np.sum( np.power(A, 2), axis=1 ) ).reshape((A.shape[0], ))

	# 	r = cp.Variable(1)
	# 	x = cp.Variable(len(self))

	# 	constraints = [A @ x + normA * r <= b]
	# 	if len(self.A_eq) > 0:
	# 		constraints += [self.A_eq @ x == self.b_eq]

	# 	problem = cp.Problem(cp.Maximize(r), constraints)
	# 	problem.solve(**self.kwargs)
	# 	radius = float(r.value)
	# 	center = np.array(x.value).reshape(len(self))

	# 	#AA = np.hstack(( A, normA.reshape(-1,1) ))
	# 	#c = np.zeros((A.shape[1]+1,))
	# 	#c[-1] = -1.0
	# 	#A_eq = np.hstack([self.A_eq, np.zeros( (self.A_eq.shape[0],1))])
	# 	#zc = linprog(c, A_ub = AA, b_ub = b, A_eq = A_eq, b_eq = self.b_eq )
	# 	#center = zc[:-1].reshape((n,))
	# 	#radius = zc[-1]
	# 	#print(center)
	# 	#print(self.isinside(center))
	# 	#zc = cp.Variable(len(self)+1)
	# 	#prob = cp.Problem(cp.Maximize(zc[-1]), [zc.__rmatmul__(AA) <= b])
	# 	#prob.solve()
	# 	#print(zc.value[0:-1])

	# 	self._radius = radius
	# 	self._cheb_center = center

	# 	return center, radius

	@property
	def radius(self):
		try:
			return self._radius
		except:
			self.chebyshev_center()
			return self._radius

	@property
	def center(self):
		try:
			return self._cheb_center
		except:
			self.chebyshev_center()
			return self._cheb_center

	@property
	def Ls(self): return []

	@property
	def ys(self): return []

	@property
	def rhos(self): return []
