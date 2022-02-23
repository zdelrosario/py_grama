"""Ridge function approximation from function values"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np
import scipy.linalg
import scipy.special
import cvxpy as cp
import warnings
from copy import deepcopy, copy

from .domains import Domain, BoxDomain
from .function import BaseFunction
from .subspace import SubspaceBasedDimensionReduction
from .ridge import RidgeFunction
from .basis import *
from .gn import gauss_newton 
from .seqlp import sequential_lp
from .exceptions import UnderdeterminedException
from .initialization import initialize_subspace
from .poly import PolynomialFunction


class PolynomialRidgeFunction(RidgeFunction):
	r""" A polynomial ridge function
	"""
	def __init__(self, basis, coef, U):
		self.basis = basis
		self.coef = np.copy(coef)
		self._U = np.array(U)
		self.domain = None

	
	def V(self, X, U = None):
		if U is None: U = self.U
		X = np.array(X)	
		Y = (U.T @ X.T).T
		return self.basis.V(Y)

	def DV(self, X, U = None):
		if U is None: U = self.U
		
		Y = (U.T @ X.T).T
		return self.basis.DV(Y)

	def DDV(self, X, U = None):
		if U is None: U = self.U
		Y = (U.T @ X.T).T
		return self.basis.DDV(Y)

	def eval(self, X):
		Vc = self.V(X) @ self.coef
		return Vc
	
	def grad(self, X):
		if len(X.shape) == 1:
			one_d = True
			X = X.reshape(1,-1)	
		else:
			one_d = False	
		
		DV = self.DV(X)
		# Compute gradient on projected space
		Df = np.tensordot(DV, self.coef, axes = (1,0))
		# Inflate back to whole space
		Df = Df.dot(self.U.T)
		if one_d:
			return Df.reshape(X.shape[1])
		else:
			return Df

	def hessian(self, X):
		if len(X.shape) == 1:
			one_d = True
			X = X.reshape(1,-1)	
		else:
			one_d = False
	
		DDV = self.DDV(X)
		DDf = np.tensordot(DDV, self.coef, axes = (1,0))
		# Inflate back to proper dimensions
		DDf = np.tensordot(np.tensordot(DDf, self.U, axes = (2,1)) , self.U, axes = (1,1)) 
		if one_d:
			return DDf.reshape(X.shape[1], X.shape[1])
		else:
			return DDf

	@property
	def profile(self):
		return PolynomialFunction(self.basis, self.coef)
	


################################################################################
# Two types of custom errors raised by PolynomialRidgeApproximation
################################################################################
class UnderdeterminedException(Exception):
	pass

class IllposedException(Exception):
	pass


def orth(U):
	""" Orthgonalize, but keep directions"""
	U, R = np.linalg.qr(U, mode = 'reduced')
	U = np.dot(U, np.diag(np.sign(np.diag(R)))) 
	return U

def inf_norm_fit(A, b):
	r""" Solve inf-norm linear optimization problem

	.. math::

		\min_{x} \| \mathbf{A} \mathbf{x} - \mathbf{b}\|_\infty

	"""
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', PendingDeprecationWarning)
		x = cp.Variable(A.shape[1])
		obj = cp.norm_inf(A @ x - b.flatten())
		problem = cp.Problem(cp.Minimize(obj))
		problem.solve(solver = 'ECOS')
		return x.value

def one_norm_fit(A, b):
	r""" solve 1-norm linear optimization problem

	.. math::

		\min_{x} \| \mathbf{a} \mathbf{x} - \mathbf{b}\|_1

	"""
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', PendingDeprecationWarning)
		x = cp.Variable(A.shape[1])
		obj = cp.norm1(x.__rmatmul__(A) - b)
		problem = cp.Problem(cp.Minimize(obj))
		problem.solve(solver = 'ECOS')
		return x.value

def two_norm_fit(A,b):
	r""" solve 2-norm linear optimization problem

	.. math::

		\min_{x} \| \mathbf{A} \mathbf{x} - \mathbf{b}\|_2

	"""
	return scipy.linalg.lstsq(A, b)[0]

def bound_fit(A, b, norm = 2):
	r""" solve a norm constrained problem

	.. math:: 

		\min_{x} \| \mathbf{A}\mathbf{x} - \mathbf{b}\|_p
		\text{such that} \mathbf{A}	\mathbf{x} -\mathbf{b} \ge 0
	"""
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', PendingDeprecationWarning)
		x = cp.Variable(A.shape[1])
		residual = x.__rmatmul__(A) - b
		if norm == 1:
			obj = cp.norm1(residual)
		elif norm == 2:
			obj = cp.norm(residual)
		elif norm == np.inf:
			obj = cp.norm_inf(residual)
		constraint = [residual >= 0] 
		#constraint = [x.__rmatmul__(A) - b >= 0]
		problem = cp.Problem(cp.Minimize(obj), constraint)
		problem.solve(feastol = 1e-10, solver = cp.ECOS)
		#problem.solve(eps = 1e-10, solver = cp.SCS)
		#problem.solve(feastol = 1e-10, solver = cp.CVXOPT)
		# TODO: The solution doesn't obey the constraints for 1 and inf norm, but does for 2-norm.
		return x.value


class PolynomialRidgeApproximation(PolynomialRidgeFunction):
	r""" Constructs a ridge approximation using a total degree approximation

	Given a basis of total degree polynomials :math:`\lbrace \psi_j \rbrace_{j=1}^N`
	on :math:`\mathbb{R}^n`, this class constructs a polynomial ridge function 
	that minimizes the mismatch on a set of points :math:`\lbrace \mathbf{x}_i\rbrace_{i=1}^M \subset \mathbb{R}^m`
	in a :math:`p`-norm:

	.. math::

		\min_{\mathbf{U} \in \mathbb{R}^{m\times n}, \  \mathbf{U}^\top \mathbf{U} = \mathbf I, \
			\mathbf{c}\in \mathbb{R}^N }
			\sqrt[p]{ \sum_{i=1}^M  \left|f(\mathbf{x}_i) - 
				\sum_{j=1}^N c_j \psi_j(\mathbf{U}^\top \mathbf{x}_i) \right|^p}

	This approach assumes :math:`\mathbf{U}` is an element of the Grassmann manifold
	obeying the orthogonality constraint.  

	For the 2-norm (:math:`p=2`) this implementation uses Variable Projection following [HC18]_ 
	to remove the solution of the linear coefficients :math:`\mathbf{c}`,
	leaving an optimization problem posed over the Grassmann manifold alone.

	For both the 1-norm and the :math:`\infty`-norm,
	this implementation uses a sequential linear program with a trust region
	coupled with a nonlinear trajectory through the search space.

	Parameters
	----------
	degree: int
		Degree of polynomial

	subspace_dimension: int
		Dimension of the low-dimensional subspace associated with the ridge approximation.
	
	basis: ['legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']
		Basis for polynomial representation
	
	norm: [1, 2, np.inf, 'inf']
		Norm in which to evaluate the mismatch between the ridge approximation and the data	
	
	scale: bool (default:True)
		Scale the coordinates along the ridge to ameliorate ill-conditioning		
	
	bound: [None, 'lower', 'upper']
		If 'lower' or 'upper' construct a lower or upper bound

	rotate: bool
		If True, rotate the U matrix to align to the active subspace with average increasing gradients

	References
	----------
	.. [HC18] J. M. Hokanson and Paul G. Constantine. 
		Data-driven Polynomial Ridge Approximation Using Variable Projection. 
		SIAM J. Sci. Comput. Vol 40, No 3, pp A1566--A1589, DOI:10.1137/17M1117690.
	"""

	def __init__(self, degree, subspace_dimension, basis = 'legendre', 
		norm = 2, n_init = 1, scale = True, keep_data = True, domain = None,
		bound = None, rotate = True, **kwargs):

		self.kwargs = kwargs
		self.rotate = rotate
		assert isinstance(degree, int)
		assert degree >= 0
		self.degree = degree
			
		assert isinstance(subspace_dimension, int)
		assert subspace_dimension >= 1
		self.subspace_dimension = subspace_dimension

		if self.degree == 1 and subspace_dimension > 1:
			self.subspace_dimension = 1
		
		if self.degree == 0:
			self.subspace_dimension = 0

		basis = basis.lower()
		assert basis in ['arnoldi', 'legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']
		self.basis_name = copy(basis)
		
		if basis == 'arnoldi':
			self.Basis = ArnoldiPolynomialBasis
		elif basis == 'legendre':
			self.Basis = LegendreTensorBasis
		elif basis == 'monomial':
			self.Basis = MonomialTensorBasis 
		elif basis == 'chebyshev':
			self.Basis = ChebyshevTensorBasis 
		elif basis == 'laguerre':
			self.Basis = LaguerreTensorBasis 
		elif basis == 'hermite':
			self.Basis = HermiteTensorBasis 

		assert isinstance(keep_data, bool)
		self.keep_data = keep_data

		assert isinstance(scale, bool)
		self.scale = scale

		assert norm in [1,2,'inf', np.inf], "Invalid norm specified"
		if norm == 'inf': norm = np.inf
		self.norm = norm

		if domain is None:
			self.domain = None
		else:
			assert isinstance(domain, Domain)
			self.domain = deepcopy(domain)


		assert bound in [None, 'lower', 'upper'], "Invalid bound specified"
		self.bound = bound

	def __len__(self):
		return self.U.shape[0]

	def __str__(self):
		return "<PolynomialRidgeApproximation degree %d, subspace dimension %d>" % (self.degree, self.subspace_dimension)


	def fit_fixed_subspace(self, X, fX, U):
		r"""

		"""
		assert U.shape[0] == X.shape[1], "U has %d rows, expected %d based on X" % (U.shape[0], X.shape[1])
		assert U.shape[1] == self.subspace_dimension, "U has %d columns; expected %d" % (U.shape[1], self.subspace_dimension)
		self._finish(X, fX, U)
		

	def fit(self, X, fX, U0 = None):
		r""" Given samples, fit the polynomial ridge approximation.

		Parameters
		----------
		X : array-like (M, m)
			Input coordinates
		fX : array-like (M,)
			Evaluations of the function at the samples
		
		"""
		kwargs = self.kwargs

		X = np.array(X)
		fX = np.array(fX).flatten()	

		assert X.shape[0] == fX.shape[0], "Dimensions of input do not match"

		# Check if we have enough data to make problem overdetermined
		m = X.shape[1]
		n = self.subspace_dimension
		d = self.degree
		n_param = scipy.special.comb(n+d, d)	# Polynomial contribution
		n_param += m*n - (n*(n+1))//2			# Number of parameters in Grassmann manifold
		if len(fX) < n_param:
			mess = "A polynomial ridge approximation of degree %d and subspace dimension %d of a %d-dimensional function " % (d, n, m)
			mess += "requires at least %d samples to not be underdetermined" % (n_param, )
			raise UnderdeterminedException(mess) 	

		# Special case where solution is convex and no iteration is required
		if self.subspace_dimension == 1 and self.degree == 1:
			self._U = self._fit_affine(X, fX)	
			self.coef = self._fit_coef(X, fX, self._U)	
			return 


		if U0 is not None:
			# Check that U0 has the right shape
			U0 = np.array(U0)
			assert U0.shape[0] == X.shape[1], "U0 has %d rows, expected %d based on X" % (U0.shape[0], X.shape[1])
			assert U0.shape[1] == self.subspace_dimension, "U0 has %d columns; expected %d" % (U0.shape[1], self.subspace_dimension)
		else:
			U0 = initialize_subspace(X = X, fX = fX)[:,:self.subspace_dimension]
			
		# Orthogonalize just to make sure the starting value satisfies constraints	
		U0 = orth(U0)
			
		# TODO Implement multiple initializations
		if self.norm == 2 and self.bound == None:
			return self._fit_varpro(X, fX, U0, **kwargs)
		else:	
			return self._fit_alternating(X, fX, U0, **kwargs)


	################################################################################	
	# Specialized Affine fits
	def _fit_affine(self, X, fX):
		r""" Solves the affine 
		"""
		# Normalize the domain 
		lb = np.min(X, axis = 0)
		ub = np.max(X, axis = 0)
		dom = BoxDomain(lb, ub) 
		XX = np.hstack([dom.normalize(X), np.ones((X.shape[0],1))])

		# Normalize the output
		fX = (fX - np.min(fX))/(np.max(fX) - np.min(fX))

		if self.bound is None:
			if self.norm == 1:
				b = one_norm_fit(XX, fX)
			elif self.norm == 2:
				b = two_norm_fit(XX, fX)
			elif self.norm == np.inf:
				b = inf_norm_fit(XX, fX)
		elif self.bound == 'lower':
			# fX >= XX b
			b = bound_fit(XX, fX, norm = self.norm)
		elif self.bound == 'upper':
			b = bound_fit(-XX, -fX, norm = self.norm)	 	

		U = b[0:-1].reshape(-1,1)
		# Correct for transform 
		U = dom._normalize_der().dot(U)
		# Force to have unit norm
		U /= np.linalg.norm(U)
		return U	

	def _fit_coef(self, X, fX, U):
		r""" Returns the linear coefficients
		"""
		Y = (U.T @ X.T).T
		self.basis = self.Basis(self.degree, X = Y) 
		V = self.basis.V(Y)
		if self.bound is None:
			if self.norm == 1:
				c = one_norm_fit(V, fX)
			elif self.norm == 2:
				c = two_norm_fit(V, fX)
			elif self.norm == np.inf:
				c = inf_norm_fit(V, fX)
			else:
				raise NotImplementedError
		elif self.bound == 'lower':
			c = bound_fit(-V, -fX, norm = self.norm)
		elif self.bound == 'upper':
			c = bound_fit(V, fX, norm = self.norm)
		
		return c
	
	def _finish(self, X, fX, U):
		r""" Given final U, rotate and find coefficients
		"""

		Y = (U.T @ X.T).T
		# Step 1: Apply active subspaces to the profile function at samples X
		# to rotate onto the most important directions
		if U.shape[1] > 1 and self.rotate:
			self._U = U
			self.coef = self._fit_coef(X, fX, U)
			grads = self.profile.grad(Y)
			# We only need the short-form SVD
			Ur = scipy.linalg.svd(grads.T, full_matrices = False)[0]
			U = U @ Ur
		
		self._U = U

		# Step 2: Flip signs such that average slope is positive in the coordinate directions
		if self.rotate:
			self.coef = self._fit_coef(X, fX, U)
			grads = self.profile.grad(Y)
			self._U = U = U.dot(np.diag(np.sign(np.mean(grads, axis = 0))))
		
		# Step 3: final fit	
		self.coef = self._fit_coef(X, fX, U)

	################################################################################	
	# VarPro based solution for the 2-norm without bound constraints 
	################################################################################	
	
	def _varpro_residual(self, X, fX, U_flat):
		U = U_flat.reshape(X.shape[1],-1)

		#V = self.V(X, U)
		Y = (U.T @ X.T).T
		self.basis = self.Basis(self.degree, Y)
		V = self.basis.V(Y)
		if self.basis_name == 'arnoldi':
			# In this case, V is orthonormal
			c = V.T @ fX
		else:
			c = scipy.linalg.lstsq(V, fX)[0].flatten()
		r = fX - V.dot(c)
		return r
	
	def _varpro_jacobian(self, X, fX, U_flat):
		# Get dimensions
		M, m = X.shape
		U = U_flat.reshape(X.shape[1],-1)
		m, n = U.shape
	
		Y = (U.T @ X.T).T
		self.basis = self.Basis(self.degree, Y)
		V = self.basis.V(Y)
		DV = self.basis.DV(Y)

		if isinstance(self.basis, ArnoldiPolynomialBasis):
			# In this case, V is orthonormal
			c = V.T @ fX
			Y = np.copy(V)
			s = np.ones(V.shape[1])
			ZT = np.eye(V.shape[1])
		else:
			c = scipy.linalg.lstsq(V, fX)[0].flatten()
			Y, s, ZT = scipy.linalg.svd(V, full_matrices = False) 

		r = fX - V.dot(c)
	
	
		N = V.shape[1]
		J1 = np.zeros((M,m,n))
		J2 = np.zeros((N,m,n))

		for ell in range(n):
			for k in range(m):
				DVDU_k = X[:,k,None]*DV[:,:,ell]
				
				# This is the first term in the VARPRO Jacobian minus the projector out fron
				J1[:, k, ell] = DVDU_k.dot(c)
				# This is the second term in the VARPRO Jacobian before applying V^-
				J2[:, k, ell] = DVDU_k.T.dot(r) 

		# Project against the range of V
		J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, (1,0)), (1,0))
		# Apply V^- by the pseudo inverse
		J2 = np.tensordot(np.diag(1./s),np.tensordot(ZT, J2, (1,0)), (1,0))
		J = -( J1 + np.tensordot(Y, J2, (1,0)))
		return J.reshape(J.shape[0], -1)
	
	def _grassmann_trajectory(self, U_flat, Delta_flat, t):
		Delta = Delta_flat.reshape(-1, self.subspace_dimension)
		U = U_flat.reshape(-1, self.subspace_dimension)
		Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		UZ = np.dot(U, ZT.T)
		U_new = np.dot(UZ, np.diag(np.cos(s*t))) + np.dot(Y, np.diag(np.sin(s*t)))
		U_new = orth(U_new).flatten()
		return U_new
	
	def _fit_varpro(self, X, fX, U0, **kwargs):
	

		def gn_solver(J_flat, r):
			Y, s, ZT = scipy.linalg.svd(J_flat, full_matrices = False, lapack_driver = 'gesvd')
			# Apply the pseudoinverse
			n = self.subspace_dimension
			Delta_flat = -ZT[:-n**2,:].T.dot(np.diag(1/s[:-n**2]).dot(Y[:,:-n**2].T.dot(r)))
			return Delta_flat, s[:-n**2]

		def jacobian(U_flat):
			return self._varpro_jacobian(X, fX, U_flat)

		def residual(U_flat):
			return self._varpro_residual(X, fX, U_flat)	

		U0_flat = U0.flatten() 
		U_flat, info = gauss_newton(residual, jacobian, U0_flat,
			trajectory = self._grassmann_trajectory, gnsolver = gn_solver, **kwargs) 
		
		U = U_flat.reshape(-1, self.subspace_dimension)
		
		self._finish(X, fX, U)	

	################################################################################	
	# Generic residual and Jacobian
	################################################################################	

	def _residual(self, X, fX, U_c):
		M, m = X.shape
		N = len(self.basis)
		n = self.subspace_dimension
		
		# Extract U and c
		U = U_c[:m*n].reshape(m,n)
		c = U_c[m*n:].reshape(N)
		
		# Construct basis
		#V = self.V(X, U)
		Y = (U.T @ X.T).T
		V = self.basis.V(Y) 
		res = V @ c - fX
		return res

	def _jacobian(self, X, fX, U_c):
		M, m = X.shape
		N = len(self.basis)
		n = self.subspace_dimension
		
		# Extract U and c
		U = U_c[:m*n].reshape(m,n)
		c = U_c[m*n:].reshape(N)

		# Re-initialize basis
		Y = (U.T @ X.T).T
		self.basis = self.Basis(self.degree, Y)
		V = self.basis.V(Y)

		# Derivative of V with respect to U with c fixed	
		DVDUc = np.zeros((M,m,n))
		#DV = self.DV(X, U) 	# Size (M, N, n)
		DV = self.basis.DV(Y)
		for k in range(m):
			for ell in range(n):
				DVDUc[:,k,ell] = X[:,k]*np.dot(DV[:,:,ell], c)
		
		# Derivative with respect to linear component
		#V = self.V(X, U)

		# Total Jacobian
		jac = np.hstack([DVDUc.reshape(M,-1), V])
		return jac

		
	def _trajectory(self, X, fX, U_c, pU_pc, alpha):
		r""" For the trajectory through the sup-norm space, we automatically compute optimal c
		and advance U along the geodesic

		"""
		M, m = X.shape
		N = len(self.basis)
		n = self.subspace_dimension
		
		# Split components
		U = orth(U_c[:m*n].reshape(m,n))
		c = U_c[m*n:].reshape(N)

		Delta = pU_pc[:m*n].reshape(m,n)
		pc = pU_pc[m*n:].reshape(N)
	
		# Orthogonalize	
		Delta = Delta - U.dot(U.T.dot(Delta))

		# Compute the step along the Geodesic	
		Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		U_new = np.dot(np.dot(U,ZT.T), np.diag(np.cos(s*alpha))) + np.dot(Y, np.diag(np.sin(s*alpha)))

		# TODO: align U and U_new to minimize Frobenius norm error 
		# right the small step termination criteria is never triggering because U_new and U have different orientations

		# Solve a convex problem to actually compute optimal c
		c = self._fit_coef(X, fX, U_new)
 
		return np.hstack([U_new.flatten(), c.flatten()])		
			
	def _fit_alternating(self, X, fX, U0, **kwargs):
		M, m = X.shape
		n = self.subspace_dimension
	
		def residual(U_c):
			r = self._residual(X, fX, U_c)
			return r
		
		def jacobian(U_c):
			m = X.shape[1]
			n = self.subspace_dimension
			U = U_c[:m*n].reshape(m,n)
			#self.set_scale(X, U)
			J = self._jacobian(X, fX, U_c)
			return J

		# Trajectory
		trajectory = lambda U_c, p, alpha: self._trajectory(X, fX, U_c, p, alpha)

		# Initialize parameter values
		#self.set_scale(X, U0)
		c0 = self._fit_coef(X, fX, U0)
		U_c0 = np.hstack([U0.flatten(), c0])

		# Add orthogonality constraints to search direction
		# Recall pU.T @ U == 0 is a requirement for Grassmann optimization
		def search_constraints(U_c, pU_pc):
			M, m = X.shape
			N = len(self.basis)
			n = self.subspace_dimension
			U = U_c[:m*n].reshape(m,n)
			constraints = [ pU_pc[k*m:(k+1)*m].__rmatmul__(U.T) == np.zeros(n) for k in range(n)]
			return constraints

		# setup lower/upper bound into SLP solver
		obj_lb = None
		obj_ub = None
		if self.bound == 'lower':
			obj_ub = np.zeros(fX.shape)
		elif self.bound == 'upper':
			obj_lb = np.zeros(fX.shape)

		# Perform optimization
		U_c = sequential_lp(residual, U_c0, jacobian, trajectory = trajectory,
			obj_lb = obj_lb, obj_ub = obj_ub,
			search_constraints = search_constraints, norm = self.norm, **kwargs)	
	
		# Store solution	
		U = U_c[:m*n].reshape(m,n)
		self._finish(X, fX, U)
	

