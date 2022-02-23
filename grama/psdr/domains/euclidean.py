import numpy as np

from scipy.stats import ortho_group
from scipy.linalg import orth
from scipy.spatial.distance import pdist

from functools import lru_cache


try:
	from functools import cached_property
except ImportError:
	from backports.cached_property import cached_property


from .domain import Domain, TOL
from ..exceptions import SolverError, EmptyDomainException, UnboundedDomainException
from ..misc import merge
from ..quadrature import gauss


class EuclideanDomain(Domain):
	r""" Abstract base class for a Euclidean input domain

	This specifies a domain :math:`\mathcal{D}\subset \mathbb{R}^m`.

	"""

	def __init__(self, names = None):
		self._init_names(names)


	################################################################################
	# Naming of variables
	################################################################################

	@property
	def names(self):
		r""" Names associated with each of the variables (coordinates) of the domain
		"""
		try:
			return self._names
		except AttributeError:
			self._names = ['x%d' % i for i in range(len(self))]
			return self._names

	def _init_names(self, names):
		if names is None:
			return

		if isinstance(names, str):
			if len(self) == 1:
				self._names = [names]
			else:
				self._names = [names + ' %d' % (j+1,) for j in range(len(self)) ]
		else:
			assert len(self) == len(names), "Number of names must match dimension"
			self._names = names

	################################################################################
	# Properties of the domain
	################################################################################

	def __len__(self):
		r""" The dimension of the Euclidean space in which this domain lives.

		Returns
		-------
		m: int
			If the domain :math:`\mathcal{D} \subset \mathbb{R}^m`, this returns
			:math:`m`.
		"""
		return self._dimension




	def normalize(self, X):
		""" Given a points in the application space, convert it to normalized units

		Parameters
		----------
		X: np.ndarray((M,m))
			points in the domain to normalize
		"""
		try:
			X.shape
		except AttributeError:
			X = np.array(X)
		if len(X.shape) == 1:
			X = X.reshape(-1, len(self))
			return self._normalize(X).flatten()
		else:
			return self._normalize(X)

	def unnormalize(self, X_norm):
		""" Convert points from normalized units into application units

		Parameters
		----------
		X_norm: np.ndarray((M,m))
			points in the normalized domain to convert to the application domain

		"""
		if len(X_norm.shape) == 1:
			X_norm = X_norm.reshape(-1, len(self))
			return self._unnormalize(X_norm).flatten()
		else:
			return self._unnormalize(X_norm)


	def normalized_domain(self, **kwargs):
		""" Return a domain with units normalized corresponding to this domain
		"""
		return self._normalized_domain(**kwargs)


	################################################################################
	# Simple properties
	################################################################################

	@property
	def lb(self): return -np.inf*np.ones(len(self))

	@property
	def ub(self): return np.inf*np.ones(len(self))

	@property
	def A(self): return np.zeros((0, len(self)))

	@property
	def b(self): return np.zeros((0,))

	@property
	def A_aug(self):
		r""" Linear inequalities augmented with bound constraints as well
		"""
		I = np.eye(len(self))
		Ilb = np.isfinite(self.lb)
		Iub = np.isfinite(self.ub)
		return np.vstack([self.A, -I[Ilb,:], I[Iub,:]])

	@property
	def b_aug(self):
		Ilb = np.isfinite(self.lb)
		Iub = np.isfinite(self.ub)
		return np.hstack([self.b, -self.lb[Ilb], self.ub[Iub]])

	@property
	def A_eq(self): return np.zeros((0, len(self)))

	@property
	def b_eq(self): return np.zeros((0,))

	@property
	def Ls(self): return []

	@property
	def ys(self): return []

	@property
	def rhos(self): return []

	@property
	def lb_norm(self):
		lb_norm = self.normalize(self.lb)
		I = ~np.isfinite(self.lb)
		lb_norm[I] = -np.inf
		return lb_norm

	@property
	def ub_norm(self):
		ub_norm = self.normalize(self.ub)
		I = ~np.isfinite(self.ub)
		ub_norm[I] = np.inf
		return ub_norm

	@property
	def A_norm(self):
		D = self._unnormalize_der()
		return self.A.dot(D)

	@property
	def b_norm(self):
		c = self._center()
		return self.b - self.A.dot(c)

	@property
	def A_eq_norm(self):
		D = self._unnormalize_der()
		return self.A_eq.dot(D)

	@property
	def b_eq_norm(self):
		c = self._center()
		return self.b_eq - self.A_eq.dot(c)

	@property
	def Ls_norm(self):
		D = self._unnormalize_der()
		return [ L.dot(D) for L in self.Ls]

	@property
	def ys_norm(self):
		c = self._center()
		return [y - c for y in self.ys]

	@property
	def rhos_norm(self):
		return self.rhos

	################################################################################
	# Meta properties
	################################################################################

	def _is_linquad_domain(self):
		return True

	def _is_linineq_domain(self):
		return len(self.Ls) == 0

	def _is_box_domain(self):
		return len(self.Ls) == 0 and len(self.b) == 0 and len(self.b_eq) == 0


	# These are the lower and upper bounds to use for normalization purposes;
	# they do not add constraints onto the domain.
	@property
	def norm_lb(self):
		r"""Lower bound used for normalization purposes; does not constrain the domain
		"""
		try:
			return self._norm_lb
		except AttributeError:
			self._norm_lb = -np.inf*np.ones(len(self))
			for i in range(len(self)):
				ei = np.zeros(len(self))
				ei[i] = 1
				if np.isfinite(self.lb[i]):
					self._norm_lb[i] = self.lb[i]
					# This ensures normalization maintains positive orientation
					if np.isfinite(self.ub[i]):
						self._norm_lb[i] = min(self._norm_lb[i], self.ub[i])
				else:
					try:
						x_corner = self.corner(-ei)
						self._norm_lb[i] = x_corner[i]
					except (SolverError, UnboundedDomainException):
						self._norm_lb[i] = -np.inf

			return self._norm_lb

	@property
	def norm_ub(self):
		r"""Upper bound used for normalization purposes; does not constrain the domain
		"""
		try:
			return self._norm_ub
		except AttributeError:
			# Temporarly disable normalization

			# Note: since this will be called by corner, we need to
			# choose a reasonable value to initialize this property, which
			# will be used until the remainder of the corner calls are made
			self._norm_ub = np.inf*np.ones(len(self))
			for i in range(len(self)):
				ei = np.zeros(len(self))
				ei[i] = 1
				if np.isfinite(self.ub[i]):
					self._norm_ub[i] = self.ub[i]
					# This ensures normalization maintains positive orientation
					if np.isfinite(self.ub[i]):
						self._norm_ub[i] = max(self._norm_ub[i], self.lb[i])
				else:
					try:
						x_corner = self.corner(ei)
						self._norm_ub[i] = x_corner[i]
					except (SolverError, UnboundedDomainException):
						self._norm_ub[i] = np.inf

			return self._norm_ub

	################################################################################
	# Normalization functions
	################################################################################

	def isnormalized(self):
		return np.all( (~np.isfinite(self.norm_lb)) | (self.norm_lb == -1.) ) and np.all( (~np.isfinite(self.norm_ub)) | (self.norm_ub == 1.) )

	def _normalize_der(self):
		"""Derivative of normalization function"""

		slope = np.ones(len(self))
		I = (self.norm_ub != self.norm_lb) & np.isfinite(self.norm_lb) & np.isfinite(self.norm_ub)
		slope[I] = 2.0/(self.norm_ub[I] - self.norm_lb[I])
		return np.diag(slope)

	def _unnormalize_der(self):
		slope = np.ones(len(self))
		I = (self.norm_ub != self.norm_lb) & np.isfinite(self.norm_lb) & np.isfinite(self.norm_ub)
		slope[I] = (self.norm_ub[I] - self.norm_lb[I])/2.0
		return np.diag(slope)

	def _center(self):
		c = np.zeros(len(self))
		I = np.isfinite(self.norm_lb) & np.isfinite(self.norm_ub)
		c[I] = (self.norm_lb[I] + self.norm_ub[I])/2.0
		return c

	def _normalize(self, X):
		c = self._center()
		D = self._normalize_der()
		X_norm = D.dot( (X - c.reshape(1,-1)).T ).T
		return X_norm

	def _unnormalize(self, X_norm, **kwargs):
		c = self._center()
		Dinv = self._unnormalize_der()
		X = Dinv.dot(X_norm.T).T + c.reshape(1,-1)
		return X

	################################################################################
	# Bound checking
	################################################################################
	def _isinside_bounds(self, X, tol = None):
		X = np.array(X)
		if tol is None: tol = self.tol
		#lb_check = np.array([np.all(x >= self.lb-tol) for x in X], dtype = np.bool)
		#ub_check = np.array([np.all(x <= self.ub+tol) for x in X], dtype = np.bool)
		lb_check = np.ones(len(X), dtype = np.bool)
		ub_check = np.ones(len(X), dtype = np.bool)
		for i in range(len(self)):
			if np.isfinite(self.lb[i]):
				lb_check &= X[:,i] >= self.lb[i] - tol
			if np.isfinite(self.ub[i]):
				ub_check &= X[:,i] <= self.ub[i] + tol
		#print("bounds check", lb_check & ub_check, self.names)
		return lb_check & ub_check

	def _isinside_ineq(self, X, tol = None):
		if tol is None: tol = self.tol
		return np.array([np.all(np.dot(self.A, x) <= self.b + tol) for x in X], dtype = np.bool)

	def _isinside_eq(self, X, tol = None):
		if tol is None: tol = self.tol
		return np.array([np.all( np.abs(np.dot(self.A_eq, x) - self.b_eq) < tol) for x in X], dtype = np.bool)

	def _isinside_quad(self, X, tol = None):
		"""check that points are inside quadratic constraints"""
		if tol is None: tol = self.tol
		inside = np.ones(X.shape[0],dtype = np.bool)
		for L, y, rho in zip(self.Ls, self.ys, self.rhos):
			diff = X - np.tile(y.reshape(1,-1), (X.shape[0],1))
			Ldiff = L.dot(diff.T).T
			Ldiff_norm = np.sum(Ldiff**2,axis=1)
			inside = inside & (np.sqrt(Ldiff_norm) <= rho + tol)
		return inside


	################################################################################
	# Extent functions
	################################################################################

	def _extent_bounds(self, x, p):
		"""Check the extent from the box constraints"""
		alpha = np.inf

		# If on the boundary, the direction needs to point inside the domain
		# otherwise we cannot move
		if np.any(p[self.lb == x] < 0):
			return 0.
		if np.any(p[self.ub == x] > 0):
			return 0.

		# To prevent divide-by-zero we ignore directions we are not moving in
		I = np.nonzero(p)

		# Check upper bounds
		y = (self.ub - x)[I]/p[I]
		if np.any(y>0):
			alpha = min(alpha, np.min(y[y>0]))

		# Check lower bounds
		y = (self.lb - x)[I]/p[I]
		if np.any(y>0):
			alpha = min(alpha, np.min(y[y>0]))

		return alpha

	def _extent_ineq(self, x, p):
		""" check the extent from the inequality constraints """
		alpha = np.inf
		# positive extent
		with np.errstate(divide = 'ignore'):
			y = (self.b - np.dot(self.A, x)	)/np.dot(self.A, p)
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))

		return alpha

	def _extent_quad(self, x, p):
		""" check the extent from the quadratic constraints"""
		alpha = np.inf
		for L, y, rho in zip(self.Ls, self.ys, self.rhos):
			Lp = L.dot(p)
			Lxy = L.dot(x - y)
			# Terms in quadratic formula a alpha^2 + b alpha + c
			a = Lp.T.dot(Lp)
			b = 2*Lp.T.dot(Lxy)
			c = Lxy.T.dot(Lxy) - rho**2

			roots = np.roots([a,b,c])
			real_roots = roots[np.isreal(roots)]
			pos_roots = real_roots[real_roots>=0]
			if len(pos_roots) > 0:
				alpha = min(alpha, min(pos_roots))
		return alpha
