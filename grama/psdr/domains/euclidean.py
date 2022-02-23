import numpy as np

from scipy.stats import ortho_group
from scipy.linalg import orth
from scipy.spatial.distance import pdist

from functools import lru_cache


try:
	from functools import cached_property
except ImportError:
	from backports.cached_property import cached_property


import cvxpy as cp

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


	
	@cached_property
	def is_empty(self):
		r""" Returns True if there are no points in the domain
		"""
		try: 
			# Try to find at least one point inside the domain
			c = self.corner(np.ones(len(self)), verbose = False)
			return False
		except EmptyDomainException:
			return True
		except UnboundedDomainException:
			return False

		
	@cached_property
	def is_point(self):
		try:
			if len(self) > 1:
				U = ortho_group.rvs(len(self))
			else:
				U = np.ones((1,1))

			for u in U:
				x1 = self.corner(u)
				x2 = self.corner(-u)
				if not np.all(np.isclose(x1, x2)):
					return False

			return True
		except EmptyDomainException:
			return False
		except UnboundedDomainException:
			return False

	@cached_property
	def is_unbounded(self):
		try:
			U = ortho_group.rvs(len(self))
			for u in U:
				x1 = self.corner(u)
				x2 = self.corner(-u)
			#	if not np.all(np.isclose(x1,x2)):
			#		self._point = False
			return False
		except UnboundedDomainException:
			return True
		except EmptyDomainException:
			return False


	################################################################################
	# Primative operations on the domain
	################################################################################

	def _build_constraints(self, x):
		raise NotImplementedError

	def _build_constraints_norm(self, x_norm):
		raise NotImplementedError

	def closest_point(self, x0, L = None, **kwargs):
		r"""Given a point, find the closest point in the domain to it.

		Given a point :math:`\mathbf x_0`, find the closest point :math:`\mathbf x`
		in the domain :math:`\mathcal D` to it by solving the optimization problem

		.. math::
		
			\min_{\mathbf x \in \mathcal D} \| \mathbf L (\mathbf x - \mathbf x_0)\|_2

		where :math:`\mathbf L` is an optional weighting matrix.		

		Parameters
		----------
		x0: array-like
			Point in :math:`\mathbb R^m`  
		L: array-like, optional
			Matrix of size (p,m) to use as a weighting matrix in the 2-norm;
			if not provided, the standard 2-norm is used.
		kwargs: dict, optional
			Additional arguments to pass to the optimizer
	
		Returns
		-------
		x: np.array (m,)
			Coordinates of closest point in this domain to :math:`\mathbf x_0`

		Raises
		------
		ValueError
			When the dimensions of x0 or L do not match those of the domain
		EmptyDomainException
			When there are no points in the domain		
		SolverError
			Raised when CVXPY fails to find a solution 
		"""
		try: 
			x0 = np.array(x0).reshape(len(self))
		except ValueError:
			raise ValueError('Dimension of x0 does not match dimension of the domain')

		if L is not None:
			try: 
				L = np.array(L).reshape(-1,len(self))
			except ValueError:
				raise ValueError('The second dimension of L does not match that of the domain')

		else:
			L = np.eye(len(self))

		local_kwargs = merge(self.kwargs, kwargs) 
		return self._closest_point(x0, L = L, **local_kwargs)

	def _closest_point(self, x0, L = None, **kwargs):
		if not self.is_linquad_domain:
			raise NotImplementedError

		# Error out if we've already determined the domain is empty
		try:
			if self._empty: raise EmptyDomainException
		except AttributeError:
			pass	
	
		# First check if the point is inside; if so we can stop	
		if self.isinside(x0):
			self._empty = False
			return np.copy(x0)
		
		# Setup the problem in CVXPY
		x_norm = cp.Variable(len(self))
		constraints = self._build_constraints_norm(x_norm)
		x0_norm = self.normalize(x0)
			
		D = self._unnormalize_der() 	
		LD = L.dot(D)
		obj = cp.norm(LD @ x_norm - LD.dot(x0_norm))

		problem = cp.Problem(cp.Minimize(obj), constraints)
		problem.solve(**kwargs)
		
		if problem.status in ['infeasible']:
			self._empty = True
			raise EmptyDomainException
		elif problem.status in ['optimal', 'optimal_inaccurate']:
			return self.unnormalize(np.array(x_norm.value).reshape(len(self)))
		else:
			raise SolverError("CVXPY exited with status '%s'" % problem.status)

	
	def corner(self, p, **kwargs):
		r""" Find the point furthest in direction p inside the domain

		Given a direction :math:`\mathbf p`, find the point furthest away in that direction

		.. math::
 	
			\max_{\mathbf{x} \in \mathcal D}  \mathbf{p}^\top \mathbf{x}

		Parameters
		----------
		p: array-like (m,)
			Direction in which to search for furthest point
		kwargs: dict, optional
			Additional parameters to be passed to cvxpy solve

		Returns
		-------
		x: np.ndarray (m,)
			Point on the boundary of the domain with :math:`m` active constraints

		Raises
		------
		EmptyDomainException
			If there is no point inside the domain

		SolverError
			If the solver errors for another reason, such as ill-conditioning			
		"""
		try:
			p = np.array(p).reshape(len(self))
		except ValueError:
			raise ValueError("Dimension of search direction doesn't match the domain dimension")

		local_kwargs = merge(self.kwargs, kwargs)
		return self._corner(p, **local_kwargs)
	
	def _corner(self, p, **kwargs):
		if not self.is_linquad_domain:
			raise NotImplementedError

		# Error out if we've already determined the domain is empty
		try:
			if self._empty: raise EmptyDomainException
		except AttributeError:
			pass
	
		# Setup the problem in CVXPY	
		x_norm = cp.Variable(len(self))
		D = self._unnormalize_der() 	
		
		# p.T @ x
		if len(self) > 1:
			obj = D.dot(p).reshape(1,-1) @ x_norm
		else:
			obj = x_norm*float(D.dot(p))

		constraints = self._build_constraints_norm(x_norm)
		problem = cp.Problem(cp.Maximize(obj), constraints)
	
		
		problem.solve(**kwargs)
		if problem.status in ['infeasible']:
			#self._empty = True
			#self._unbounded = False
			#self._point = False
			raise EmptyDomainException	
		# For some reason, if no constraints are provided CVXPY doesn't note
		# the domain is unbounded
		elif problem.status in ['unbounded'] or len(constraints) == 0:
			#self._unbounded = True
			#self._empty = False
			#self._point = False
			raise UnboundedDomainException
		elif problem.status not in ['optimal', 'optimal_inaccurate']:
			raise SolverError("CVXPY exited with status '%s'" % problem.status)

		# If we have found a solution, then the domain is not empty
		self._empty = False
		return self.unnormalize(np.array(x_norm.value).reshape(len(self)))
		


	def constrained_least_squares(self, A, b, **kwargs):
		r"""Solves a least squares problem constrained to the domain

		Given a matrix :math:`\mathbf{A} \in \mathbb{R}^{n\times m}`
		and vector :math:`\mathbf{b} \in \mathbb{R}^n`,
		solve least squares problem where the solution :math:`\mathbf{x}\in \mathbb{R}^m`
		is constrained to the domain :math:`\mathcal{D}`:

		.. math::
		
			\min_{\mathbf{x} \in \mathcal{D}} \| \mathbf{A} \mathbf{x} - \mathbf{b}\|_2^2
		
		Parameters
		----------
		A: array-like (n,m)	
			Matrix in least squares problem
		b: array-like (n,)
			Right hand side of least squares problem
		kwargs: dict, optional
			Additional parameters to pass to solver
		""" 
		try:
			A = np.array(A).reshape(-1,len(self))
		except ValueError:
			raise ValueError("Dimension of matrix A does not match that of the domain")
		try:
			b = np.array(b).reshape(A.shape[0])
		except ValueError:
			raise ValueError("dimension of b in least squares problem doesn't match A")

		return self._constrained_least_squares(A, b, **kwargs)	
	
	def _constrained_least_squares(self, A, b, **kwargs):
		if not self.is_linquad_domain:
			raise NotImplementedError

		# Error out if we've already determined the domain is empty
		try:
			if self._empty: raise EmptyDomainException
		except AttributeError:
			pass
		
		# Setup the problem in CVXPY	
		x_norm = cp.Variable(len(self))
		D = self._unnormalize_der() 
		c = self._center()	
			
		# \| A x - b\|_2 
		obj = cp.norm( (A @ D) @ x_norm - b - (A @ c) )
		constraints = self._build_constraints_norm(x_norm)
		problem = cp.Problem(cp.Minimize(obj), constraints)
		problem.solve(**kwargs)
		
		if problem.status in ['infeasible']:
			self._empty = True
			self._unbounded = False
			self._point = False
			raise EmptyDomainException	
		elif problem.status in ['unbounded']:
			self._unbounded = True
			self._empty = False
			self._point = False
			raise UnboundedDomainException
		elif problem.status not in ['optimal', 'optimal_inaccurate']:
			raise SolverError("CVXPY exited with status '%s'" % problem.status)

		# If we have found a solution, then the domain is not empty
		self._empty = False
		return self.unnormalize(np.array(x_norm.value).reshape(len(self)))

		
	
	################################################################################
	# Utility functions for the domain
	################################################################################

	def sweep(self, n = 20, x = None, p = None, corner = False):
		r""" Constructs samples for a random parameter sweep


		Parameters
		----------
		n: int, optional [default: 20]
			Number of points to sample along the direction.
		x: array-like, optional [default: random location in the domain]
			Point in the domain through which the sweep runs.
		p: array-like, optional [default: random]
			Direction in which to sweep.
		corner: bool, optional 
			If true, sweep between two opposite corners rather than until the boundary is hit.

		Returns
		-------
		X: np.ndarray (n, len(self))
			Points along the parameter sweep
		y: np.ndarray (n,)
			Length along sweep	
		"""
		n = int(n)

		if x is None:
			x = self.sample()
		else:
			assert self.isinside(x), "Provided x not inside the domain"
	
		if p is None:
			# Choose a valid direction
			p = np.random.randn(len(self))
		else:
			assert len(p) == len(self), "Length of direction vector 'p' does not match the domain"

		if corner:
			# Two end points for search
			c1 = self.corner(p)
			c2 = self.corner(-p)
		else:
			# Orthogonalize search direction against equality constraints 
			Qeq = self._A_eq_basis
			p -= Qeq.dot(Qeq.T.dot(p))

			a1 = self.extent(x, p)
			c1 = x + a1*p
			a2 = -self.extent(x, -p)
			c2 = x + a2*p
		
		# Samples
		X = np.array([ (1-alpha)*c1 + alpha*c2 for alpha in np.linspace(0,1,n)])

		# line direction
		d = (X[1] - X[0])/np.linalg.norm(X[1] - X[0])	
		y = X.dot(d)
		return X, y

	@property
	def intrinsic_dimension(self):
		r""" The intrinsic dimension (ambient space minus equality constraints)"""
		return len(self) - self.A_eq.shape[0]


	


	# To define the documentation once for all domains, these functions call internal functions
	# to each subclass
	
	


	def extent(self, x, p):
		r"""Compute the distance alpha such that x + alpha * p is on the boundary of the domain

		Given a point :math:`\mathbf{x}\in\mathcal D` and a direction :math:`\mathbf p`,
		find the furthest we can go in direction :math:`\mathbf p` and stay inside the domain:

		.. math::
	
			\max_{\alpha > 0}   \alpha \quad\\text{such that} \quad \mathbf x +\\alpha \mathbf p \in \mathcal D

		Parameters
		----------
		x : np.ndarray(m)
			Starting point in the domain

		p : np.ndarray(m)
			Direction from p in which to head towards the boundary

		Returns
		-------
		alpha: float
			Distance to boundary along direction p
		"""
		try:
			x = np.array(x).reshape(len(self))
		except ValueError:
			raise ValueError("Starting point not the same dimension as the domain")

		assert self.isinside(x), "Starting point must be inside the domain" 
		return self._extent(x, p)

	def _extent(self, x, p):
		raise NotImplementedError

	def isinside(self, X, tol = TOL):
		""" Determine if points are inside the domain

		Parameters
		----------
		X : np.ndarray(M, m)
			Samples in rows of X
		"""
		# Make this a 2-d array
		X = np.atleast_1d(X)
		if len(X.shape) == 1:
			# Check for dimension mismatch
			if X.shape[0] != len(self):
				return False
			X = X.reshape(-1, len(self)) 	
			return self._isinside(X, tol = tol).flatten()
		else:
			# Check if the dimensions match
			if X.shape[1] != len(self):
				return np.zeros(X.shape[0], dtype = np.bool)
			return self._isinside(X, tol = tol)

	def _isinside(self, X, tol = TOL):
		raise NotImplementedError

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
	
	def __mul__(self, other):
		""" Combine two domains
		"""
		from .tensor import TensorProductDomain
		return TensorProductDomain([self, other])
	
	def __rmul__(self, other):
		""" Combine two domains
		"""
		from .tensor import TensorProductDomain
		return TensorProductDomain([self, other])
	

	def sample(self, draw = 1):
		""" Sample points with uniform probability from the measure associated with the domain.

		This is intended as a low-level interface for generating points from the domain.
		More advanced approaches are handled through the Sampler subclasses.

		Parameters
		----------
		draw: int
			Number of samples to return

		Returns
		-------
		array-like (draw, len(self))
			Array of samples from the domain

		Raises
		------
		SolverError
			If we are unable to find a point in the domain satisfying the constraints
		"""
		draw = int(draw)
		
		# If request a non positive number of samples, simply return zero
		if draw <= 0:
			return np.zeros((0, len(self)))

		x_sample = self._sample(draw = draw)
		if draw == 1: 
			x_sample = x_sample.flatten()
		return x_sample
	
	def _sample(self, draw = 1):
		# By default, use the hit and run sampler

		# However, we only use hit and run if it isn't a point
		if self.is_point:
			c = self.center
			return np.array([c for i in range(draw)])

		X = [self._hit_and_run() for i in range(3*draw)]
		I = np.random.permutation(len(X))
		return np.array([X[i] for i in I[0:draw]])


	def sample_grid(self, n):
		r""" Sample points from a tensor product grid inside the domain
	
		For a bounded domain this function provides samples that come from a uniformly spaced grid.
		This grid contains `n` points in each direction, linearly spaced between the lower and upper bounds.
		For box domains, this will contain $n^d$ samples where $d$ is the dimension of the domain.
		For other domains, this will potentially contain fewer samples since points on the grid outside the domain
		are excluded.
	
		Parameters
		----------
		n: int
			Number of samples in each direction
		"""

		assert np.all(np.isfinite(self.lb)) & np.all(np.isfinite(self.ub)), "Requires a bounded domain"
		xs = [np.linspace(lbi, ubi, n) for lbi, ubi in zip(self.lb, self.ub)]
		Xs = np.meshgrid(*xs, indexing = 'ij')
		Xgrid = np.vstack([X.flatten() for X in Xs]).T
		I = self.isinside(Xgrid)
		return Xgrid[I]	


	def sample_boundary(self, draw):
		r""" Sample points on the boundary of the domain

		Parameters
		----------
		draw : int
			Number of points to sample
		"""
		draw = int(draw)
	
		dirs = [self.random_direction(self.center) for i in range(draw)]
		X = np.array([self.corner(d) for d in dirs])
	
		if draw == 1:
			return X.flatten()
		
		return X
	


	def random_direction(self, x):
		r""" Returns a random direction that can be moved and still remain in the domain

		Parameters
		----------
		x: array-like
			Point in the domain
		
		Returns
		-------
		p: np.ndarray (m,)
			Direction that stays inside the domain
		"""

		if not self.is_linquad_domain:
			raise NotImplementedError

		Qeq = self._A_eq_basis
		while True:
			# Generate a random direction inside 
			p = np.random.normal(size = (len(self),))
			# Orthogonalize against equality constarints constraints
			p = p - Qeq @ (Qeq.T @ p)
			# check that the direction isn't hitting a constraint
			if x is None or self.extent(x, p) > 0:
				break
		return p	

	def quadrature_rule(self, N, method = 'auto'):
		r""" Constructs quadrature rule for the domain

		Given a maximum number of samples N, 
		this function constructs a quadrature rule for the domain
		using :math:`M \le N` samples such that 

		.. math::
		
			\int_{\mathbf x\in \mathcal D} f(\mathbb{x}) \mathrm{d}\mathbf{x}
			\approx \sum_{j=1}^M w_j f(\mathbf{x}_j).

		
		
		Parameters
		----------
		N: int
			Number of samples to use to construct estimate
		method: string, ['auto', 'gauss', 'montecarlo']
			Method to use to construct quadrature rule

		Returns
		-------
		X: np.ndarray (M, len(self))
			Samples from the domain
		w: np.ndarray (M,)
			Weights for quadrature rule

		"""
		from .normal import NormalDomain
		if isinstance(self, NormalDomain):
			if self.truncate is not None:
				raise NotImplementedError

		# If we have a single point in the domain, we can't really integrate
		if self.is_point:
			return self.sample().reshape(1,-1), np.ones(1) 
	
		N = int(N)
	
		# The number of points in each direction we could use for a tensor-product
		# quadrature rule
		q = int(np.floor( N**(1./len(self))))
		if method == 'auto':
			# If we can take more than one point in each axis, use a tensor-product Gauss quadrature rule
			if q > 1: method = 'gauss'
			else: method = 'montecarlo'

		if self.is_unbounded:
			method = 'montecarlo'

		# We currently do not support gauss quadrature on equality constrained domains
		if len(self.A_eq) > 0 and method == 'gauss':
			method = 'montecarlo'

		if method == 'gauss':

			def quad(qs):
				# Constructs a quadrature rule for the domain, restricting to those points that are inside
				xs = []
				ws = []
				for i in range(len(self)):
					x, w = gauss(qs[i], self.norm_lb[i], self.norm_ub[i])
					xs.append(x)
					ws.append(w)
				# Construct the samples	
				Xs = np.meshgrid(*xs)
				# Flatten into (M, len(self)) shape 
				X = np.hstack([X.reshape(-1,1) for X in Xs])
			
				# Construct the weights
				Ws = np.meshgrid(*ws)
				W = np.hstack([W.reshape(-1,1) for W in Ws])
				w = np.prod(W, axis = 1)
				
				# remove those points outside the domain
				I = self.isinside(X)
				X = X[I]
				w = w[I]	
			
				return X, w

			qs = q*np.ones(len(self))
			X, w = quad(qs)

			# If all the points were in the domain, stop
			if np.prod(qs) == len(X):
				return X, w

			# Now we iterate, increasing the density of the quadrature rule
			# while staying below the maximum number of points 
			# TODO: Use a bisection search to find the right spacing
			while True:
				# increase dimension of the rule 
				qs += 1.
				Xnew, wnew = quad(qs)
				if len(Xnew) <= N:
					X = Xnew
					w = wnew
				else:
					break

			return X, w 

		elif method == 'montecarlo':
			# For a Monte-Carlo rule we simply sample the domain randomly.
			w = (1./N)*np.ones(N)
			X = self.sample(N)

			# However, we need to include a correction to account for the 
			# volume of this domain
			vol = self.volume()
			w *= vol
			return X, w

	@lru_cache()
	def volume(self, N = 1e4):
		if self.is_box_domain:
			# if the domain is a box domain, this is simple
			vol = np.prod(self.ub - self.lb)
		else:
			N = int(N)
			# Otherwise we estimate the volume of domain using Monte-Carlo
			Xt = np.random.uniform(self.norm_lb, self.norm_ub, size = (N, len(self)))
			vol = np.prod(self.norm_ub - self.norm_lb)*(np.sum(self.isinside(Xt))/(N))

		return vol	


	@cached_property
	def _A_eq_basis(self):
		r""" This contains an orthogonal basis for the directions that are *not* allowed to be moved along
		"""
		try: 
			if len(self.A_eq) == 0: raise AttributeError
			Qeq = orth(self.A_eq.T)
		except AttributeError:
			Qeq = np.zeros((len(self),0))
		return Qeq


	def _corner_center(self):
		# Otherwise we pick points on the boundary and then initialize
		# at the center of the domain.

		# Generate random orthogonal directions to sample
		if len(self) > 1:
			U = ortho_group.rvs(len(self))
		else:
			U = np.ones((1,1))
		X = []
		for i in range(len(self)):
			X += [self.corner(U[:,i])]
			X += [self.corner(-U[:,i])]
			if i >= 3 and not np.isclose(np.max(pdist(X)),0):
				# If we have collected enough points and these
				# are distinct, stop
				break
			
		# If we still only have effectively one point, we are a point domain	
		if np.isclose(np.max(pdist(X)),0) and len(X) == 2*len(self):
			self._point = True
		else:
			self._point = False

		# Take the mean
		x0 = sum(X)/len(X)
		x0 = self.closest_point(x0)
		return x0

	def _hit_and_run(self, _recurse = 2):
		r"""Hit-and-run sampling for the domain
		"""
		if _recurse < 0:
			raise ValueError("Could not find valid hit and run step")

		try:
			# Get the current location of where the hit and run sampler is
			x0 = self._hit_and_run_state
			if x0 is None: raise AttributeError
		except AttributeError:
		
			try:
				# The simpliest and inexpensive approach is to start hit and run
				# at the Chebyshev center of the domain.  This value is cached 
				# and so will not require recomputation if reinitialized
				x0, r = self.chebyshev_center()
	
				# TODO: chebyshev_center breaks when running test_lipschitz_sample yielding a SolverError
				# It really shouldn't error
			except (AttributeError, NotImplementedError):
				x0 = self._corner_center()
			self._hit_and_run_state = x0
		
		
		# If we are point domain, there is no need go any further
		if self.is_point:
			return self._hit_and_run_state.copy()
	
		# See if there is an orthongonal basis for the equality constraints
		# This is necessary so we can generate random directions that satisfy the equality constraint.
		# TODO: Should we generalize this as a "tangent cone" or "feasible cone" that each domain implements?
		Qeq = self._A_eq_basis

		# Loop over multiple search directions if we have trouble 
		for it in range(len(self)):	
			p = self.random_direction(x0)
			# Orthogonalize against equality constarints constraints
			p /= np.linalg.norm(p)

			alpha_min = -self.extent(x0, -p)
			alpha_max =  self.extent(x0,  p)
			if alpha_max - alpha_min > 1e-7:
				alpha = np.random.uniform(alpha_min, alpha_max)
				# We call closest point just to make sure we stay inside numerically
				self._hit_and_run_state = self.closest_point(self._hit_and_run_state + alpha*p)
				return np.copy(self._hit_and_run_state)	
		
		# If we've failed to find a good direction, reinitialize, and recurse
		self._hit_and_run_state = None
		return self._hit_and_run(_recurse = _recurse - 1)


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
