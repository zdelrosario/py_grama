""" Sequential Linear Program
"""
from __future__ import print_function
import numpy as np
import scipy.optimize
import cvxpy as cp
import warnings

from .domains import UnboundedDomain
from .gn import trajectory_linear

class InfeasibleException(Exception):
	pass

class UnboundedException(Exception):
	pass


def sequential_lp(f, x0, jac, search_constraints = None,
	norm = 2, trajectory = trajectory_linear, obj_lb = None, obj_ub = None,
	constraints = None, constraint_grads = None, constraints_lb = None, constraints_ub = None,
	maxiter = 100, bt_maxiter = 50, domain = None,
	tol_dx = 1e-10, tol_obj = 1e-10,  verbose = False, **kwargs):
	r""" Solves a nonlinear optimization problem by a sequence of linear programs


	Given the optimization problem

	.. math::

		\min{\mathbf{x} \in \mathbb{R}^m} &\ \| \mathbf{f}(\mathbf{x}) \|_p \\
		\text{such that} & \  \text{lb} \le \mathbf{f} \le \text{ub} \\
			& \ \text{constraint_lb} \le \mathbf{g} \le \text{constraint_ub}

	this function solves this problem by linearizing both the objective and constraints
	and solving a sequence of disciplined convex problems.


	Parameters
	----------
	norm: [1,2, np.inf, None, 'hinge']
		If hinge, sum of values of the objective exceeding 0.

	
	References
	----------
	.. FS89


	"""
	assert norm in [1,2,np.inf, None, 'hinge'], "Invalid norm specified."

	if search_constraints is None:
		search_constraints = lambda x, p: []
	
	if domain is None:
		domain = UnboundedDomain(len(x0))

	if constraints is None:
		constraints = []
	if constraint_grads is None:
		constraint_grads = []

	assert len(constraints) == len(constraint_grads), "Must provide same number of constraints as constraint gradients"

	if constraints_lb is None:
		constraints_lb = -np.inf*np.ones(len(constraints))
	
	if constraints_ub is None:
		constraints_ub = np.inf*np.ones(len(constraints))


	# The default solver for 1/inf-norm doesn't converge sharp enough, but ECOS does.	
	if 'solver' not in kwargs:
		kwargs['solver'] = 'ECOS'


	if norm in [1,2, np.inf]:
		objfun = lambda fx: np.linalg.norm(fx, ord = norm)
	elif norm == 'hinge':
		objfun = lambda fx: np.sum(np.maximum(fx, 0))
	else:
		objfun = lambda fx: float(fx)

	# evalutate KKT norm
	def kkt_norm(fx, jacx):
		kkt_norm = np.nan
		if norm == np.inf:
			# TODO: allow other constraints into the solution
			t = objfun(fx)
			obj_grad = np.zeros(len(x)+1)
			obj_grad[-1] = 1.
			con = np.hstack([fx - t, -fx -t])
			con_grad = np.zeros((2*len(fx),len(x)+1))
			con_grad[:len(fx),:-1] = jacx
			con_grad[:len(fx),-1] = -1.
			con_grad[len(fx):,:-1] = -jacx
			con_grad[len(fx):,-1] = -1.

			# Find the active constraints (which have non-zero Lagrange multipliers)
			I = np.abs(con) < 1e-10
			lam, kkt_norm = scipy.optimize.nnls(con_grad[I,:].T, -obj_grad)
		elif norm == 1:
			t = np.abs(fx)
			obj_grad = np.zeros(len(x) + len(fx))
			obj_grad[len(x):] = 1.
			con = np.hstack([fx - t, -fx-t])
			con_grad = np.zeros((2*len(fx), len(x)+len(fx)))
			con_grad[:len(fx),:len(x)] = jacx
			con_grad[:len(fx),len(x):] = -1.
			con_grad[len(fx):,:len(x)] = -jacx
			con_grad[len(fx):,len(x):] = -1.
			I = np.abs(con) == 0.
			lam, kkt_norm = scipy.optimize.nnls(con_grad[I,:].T, -obj_grad)

		elif norm == 2:
			kkt_norm = np.linalg.norm(jacx.T.dot(fx))

		# TODO: Should really orthogonalize against unallowed search directions
		#err = con_grad[I,:].T.dot(lam) + obj_grad
		#print err
	
		return kkt_norm

	# Start optimizaiton loop
	x = np.copy(x0)
	try:
		fx = np.array(f(x))
	except TypeError:
		fx = np.array([fi(x) for fi in f]).reshape(-1,)

	objval = objfun(fx)

	try:
		jacx = jac(x)
	except TypeError:
		jacx = np.array([jaci(x) for jaci in jac]).reshape(len(fx), len(x))

	if verbose:
		print('iter |     objective     |  norm px | TR radius | KKT norm | violation |')
		print('-----|-------------------|----------|-----------|----------|-----------|')
		print('%4d | %+14.10e |          |           | %8.2e |           |' % (0, objval, kkt_norm(fx, jacx))) 

	Delta = 1.

	for it in range(maxiter):
	
		# Search direction
		p = cp.Variable(len(x))

		# Linearization of the objective function
		f_lin = fx + p.__rmatmul__(jacx)

		if norm == 1: obj = cp.norm1(f_lin)
		elif norm == 2: obj = cp.norm(f_lin)
		elif norm == np.inf: obj = cp.norm_inf(f_lin)
		elif norm == 'hinge': obj = cp.sum(cp.pos(f_lin))
		elif norm == None: obj = f_lin
		else: raise NotImplementedError
		# Now setup constraints
		nonlinear_constraints = []

		# First, constraints on "f"
		if obj_lb is not None:
			nonlinear_constraints.append(obj_lb <= f_lin)
		if obj_ub is not None:
			nonlinear_constraints.append(f_lin <= obj_ub)  
		
		# Next, we add other nonlinear constraints
		for con, congrad, con_lb, con_ub in zip(constraints, constraint_grads, constraints_lb, constraints_ub):
			conx = con(x)
			congradx = congrad(x)
			#print "conx", conx, congradx
			if np.isfinite(con_lb):
				nonlinear_constraints.append(con_lb <= conx + p.__rmatmul__(congradx) )
			if np.isfinite(con_ub):
				nonlinear_constraints.append(conx + p.__rmatmul__(congradx) <= con_ub )

		# Constraints on the search direction specified by user
		search_step_constraints = search_constraints(x, p)

		# Append constraints from the domain of x
		domain_constraints = domain._build_constraints(x + p)

		stop = False
		for it2 in range(bt_maxiter):
			active_constraints = nonlinear_constraints + domain_constraints + search_step_constraints

			if it2 > 0:
				trust_region_constraints = [cp.norm(p) <= Delta]
				active_constraints += trust_region_constraints

			# Solve for the search direction

			with warnings.catch_warnings():
				warnings.simplefilter('ignore', PendingDeprecationWarning)
				try:
					problem = cp.Problem(cp.Minimize(obj), active_constraints)
					problem.solve(**kwargs)
					status = problem.status
				except cp.SolverError:
					if it2 == 0:
						status = 'unbounded'
					else:
						status = cp.SolverError

			if (status == 'unbounded' or status == 'unbounded_inaccurate') and it2 == 0:
				# On the first step, the trust region is off, allowing a potentially unbounded domain
				pass
			elif status in ['optimal', 'optimal_inaccurate']:
				# Otherwise, we've found a feasible step
				px = p.value
				# Evaluate new point along the trajectory
				x_new = trajectory(x, px, 1.)

				# Check for movement of the point
				if np.all(np.isclose(x, x_new, rtol = tol_dx, atol = 0)):
					stop = True
					break

				# Evaluate value at new point
				try:
					fx_new = np.array(f(x_new))
				except TypeError:
					fx_new = np.array([fi(x_new) for fi in f]).reshape(-1,)
				objval_new = objfun(fx_new)

				constraint_violation = 0.
				if obj_lb is not None:
					I = ~(obj_lb <= fx_new)
					constraint_violation += np.linalg.norm((fx_new - obj_lb)[I], 1)
				if obj_ub is not None:
					I = ~(fx_new <= obj_ub)
					constraint_violation += np.linalg.norm((fx_new - obj_ub)[I], 1)

				if objval_new < objval and np.isclose(constraint_violation, 0., rtol = 1e-10, atol = 1e-10):
					x = x_new
					fx = fx_new

					if np.abs(objval_new - objval) < tol_obj:
						stop = True
					objval = objval_new
					Delta = max(1., np.linalg.norm(px))
					break

				Delta *=0.5
		
			else:
				warnings.warn("Could not find acceptible step; stopping prematurely; %s" % (status,) )
				stop = True
				px = np.zeros(x.shape)
				
			#elif status in ['unbounded', 'unbounded_inaccurate']:
			#	raise UnboundedException
			#elirf status in ['infeasible']:
			#	raaise InfeasibleException 
			#else:
			#	raise Exception(status)
	
		if it2 == bt_maxiter-1:
			stop = True

		# Update the jacobian information
		try:
			jacx = jac(x)
		except TypeError:
			jacx = np.array([jaci(x) for jaci in jac]).reshape(len(fx), len(x))

		if verbose:
			print('%4d | %+14.10e | %8.2e |  %8.2e | %8.2e |  %8.2e |' 
				% (it+1, objval, np.linalg.norm(px), Delta, kkt_norm(fx, jacx), constraint_violation))
		if stop:
			break	

	return x
			
#if __name__ == '__main__':
#	from polyridge import *
#	
#	np.random.seed(3)
#	p = 3
#	m = 4
#	n = 1
#	M = 100
#
#	norm = np.inf
#	norm = 2
#	U = orth(np.random.randn(m,n))
#	coef = np.random.randn(len(LegendreTensorBasis(n,p)))
#	prf = PolynomialRidgeFunction(LegendreTensorBasis(n,p), coef, U)
#
#	X = np.random.randn(M,m)
#	fX = prf.eval(X)
#
#	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension  = n, norm = norm, scale = True)
#
#	def residual(U_c):
#		r = pra._residual(X, fX, U_c)
#		return r
#
#	def jacobian(U_c):
#		U = U_c[:m*n].reshape(m,n)
#		pra.set_scale(X, U)
#		J = pra._jacobian(X, fX, U_c)
#		return J
#	
#	# Trajectory
#	trajectory = lambda U_c, p, alpha: pra._trajectory(X, fX, U_c, p, alpha)
#
#	def search_constraints(U_c, pU_pc):
#	#	M, m = X.shape
#	#	N = len(self.basis)
#	#	n = self.subspace_dimension
#		U = U_c[:m*n].reshape(m,n)
#		constraints = [ pU_pc[k*m:(k+1)*m].__rmatmul__(U.T) == np.zeros(n) for k in range(n)]
#		return constraints
#
#		
#	U0 = orth(np.random.randn(m,n))
#	U0 = U
#	c = np.random.randn(len(coef))
#	pra.set_scale(X, U0)
#	U_c0 = np.hstack([U0.flatten(), c])
#
#	U_c = sequential_lp(residual, U_c0, jacobian, search_constraints, norm = norm, 
#		trajectory = trajectory, verbose = True)
#	print(U_c)
#	print(U)	
