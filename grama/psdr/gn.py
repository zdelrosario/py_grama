# 2018 (c) Jeffrey M. Hokanson and Caleb Magruder
from __future__ import print_function, division

import warnings
import numpy as np
import scipy.linalg
import scipy as sp


__all__ = ['linesearch_armijo',
	'gauss_newton',
	'trajectory_linear',
	]


class BadStep(Exception):
	pass


def trajectory_linear(x0, p, t):
	return x0 + t * p


def linesearch_armijo(f, g, p, x0, bt_factor=0.5, ftol=1e-4, maxiter=40, trajectory = trajectory_linear, fx0 = None):
	"""Back-Tracking Line Search to satify Armijo Condition

		f(x0 + alpha*p) < f(x0) + alpha * ftol * <g,p>

	Parameters
	----------
	f : callable
		objective function, f: R^n -> R
	g : np.array((n,))
		gradient
	p : np.array((n,))
		descent direction
	x0 : np.array((n,))
		current location
	bt_factor : float [optional] default = 0.5
		backtracking factor
	ftol : float [optional] default = 1e-4
		coefficient in (0,1); see Armijo description in Nocedal & Wright
	maxiter : int [optional] default = 10
		maximum number of iterations of backtrack
	trajectory: function(x0, p, t) [Optional]
		Function that returns next iterate 
	Returns
	-------
	float
		alpha: backtracking coefficient (alpha = 1 implies no backtracking)
	"""
	
	dg = np.inner(g, p)
	assert dg <= 0, 'Descent direction p is not a descent direction: p^T g = %g >= 0' % (dg, )

	alpha = 1

	if fx0 is None:
		fx0 = f(x0)

	fx0_norm = np.linalg.norm(fx0)
	x = np.copy(x0)
	fx = np.inf
	success = False
	for it in range(maxiter):
		try:
			x = trajectory(x0, p, alpha)
			fx = f(x)
			fx_norm = np.linalg.norm(fx)
			if fx_norm < fx0_norm + alpha * ftol * dg:
				success = True
				break
		except BadStep:
			pass
			
		alpha *= bt_factor

	# If we haven't found a good step, stop
	if not success:
		alpha = 0
		x = x0
		fx = fx0
	return x, alpha, fx


def gauss_newton(f, F, x0, tol=1e-10, tol_normdx=1e-12, 
	maxiter=100, linesearch=None, verbose=0, trajectory=None, gnsolver = None):
	r"""A Gauss-Newton solver for unconstrained nonlinear least squares problems.

	Given a vector valued function :math:`\mathbf{f}:\mathbb{R}^m \to \mathbb{R}^M`
	and its Jacobian :math:`\mathbf{F}:\mathbb{R}^m\to \mathbb{R}^{M\times m}`,
	solve the nonlinear least squares problem:

	.. math::

		\min_{\mathbf{x}\in \mathbb{R}^m} \| \mathbf{f}(\mathbf{x})\|_2^2.

	Normal Gauss-Newton computes a search direction :math:`\mathbf{p}\in \mathbb{R}^m`
	at each iterate by solving least squares problem

	.. math::
		
		\mathbf{p}_k \leftarrow \mathbf{F}(\mathbf{x}_k)^+ \mathbf{f}(\mathbf{x}_k)

	and then computes a new step by solving a line search problem for a step length :math:`\alpha`
	satisfying the Armijo conditions:

	.. math::
		
		\mathbf{x}_{k+1} \leftarrow \mathbf{x}_k + \alpha \mathbf{p}_k.
		
	This implementation offers several features that modify this basic outline.

	First, the user can specify a nonlinear *trajectory* along which candidate points 
	can move; i.e.,

	.. math::

		\mathbf{x}_{k+1} \leftarrow T(\mathbf{x}_k, \mathbf{p}_k, \alpha). 
	
	Second, the user can specify a custom solver for computing the search direction :math:`\mathbf{p}_k`.

	Parameters
	----------
	f : callable
		residual, :math:`\mathbf{f}: \mathbb{R}^m \to \mathbb{R}^M`
	F : callable
		Jacobian of residual :math:`\mathbf{f}`; :math:`\mathbf{F}: \mathbb{R}^m \to \mathbb{R}^{M \times m}`
	tol: float [optional] default = 1e-8
		gradient norm stopping criterion
	tol_normdx: float [optional] default = 1e-12
		norm of control update stopping criterion
	maxiter : int [optional] default = 100
		maximum number of iterations of Gauss-Newton
	linesearch: callable, returns new x
		f : callable, residual, f: R^n -> R^m
		g : gradient, R^n
		p : descent direction, R^n
		x0 : current iterate, R^n
	gnsolver: [optional] callable, returns search direction p 
		Parameters: 
			F: current Jacobian
			f: current residual

		Returns:
			p: search step
			s: singular values of Jacobian
	verbose: int [optional] default = 0
		if >= print convergence history diagnostics

	Returns
	-------
	numpy.array((dof,))
		returns x^* (optimizer)
	int
		info = 0: converged with norm of gradient below tol
		info = 1: norm of gradient did not converge, but ||dx|| below tolerance
		info = 2: did not converge, max iterations exceeded
	"""
	n = len(x0)
	if maxiter <= 0: return x0, 4

	if verbose >= 1:
		print('Gauss-Newton Solver Iteration History')
		print('  iter   |   ||f(x)||   |   ||dx||   | cond(F(x)) |   alpha    |  ||grad||  ')
		print('---------|--------------|------------|------------|------------|------------')
	if trajectory is None:
		trajectory = lambda x0, p, t: x0 + t * p

	if linesearch is None:
		linesearch = linesearch_armijo
			
	if gnsolver is None:
		# Scipy seems to properly check for proper allocation of working space, reporting an error with gelsd
		# so we specify using gelss (an SVD based solver)
		def gnsolver(F_eval, f_eval):
			dx, _, _, s = sp.linalg.lstsq(F_eval, -f_eval, lapack_driver = 'gelss')
			return dx, s

	x = np.copy(x0)
	f_eval = f(x)
	F_eval = F(x)
	grad = F_eval.T @ f_eval

	normgrad = np.linalg.norm(grad)

	#rescale tol by norm of initial gradient
	tol = max(tol*normgrad, 1e-14)

	normdx = 1
	for it in range(maxiter):
		residual_increased = False
		
		# Compute search direction
		dx, s = gnsolver(F_eval, f_eval)
		
		# Check we got a valid search direction
		if not np.all(np.isfinite(dx)):
			raise RuntimeError("Non-finite search direction returned") 
		
		# If Gauss-Newton step is not a descent direction, use -gradient instead
		if np.inner(grad, dx) >= 0:
			dx = -grad
		
		# Back tracking line search
		x_new, alpha, f_eval_new = linesearch(f, grad, dx, x, trajectory=trajectory)
		

		normf = np.linalg.norm(f_eval_new)	
		if np.linalg.norm(f_eval_new) >= np.linalg.norm(f_eval):
			residual_increased = True
		else:
			#f_eval = f(x)
			f_eval = f_eval_new
			x = x_new

		normdx = np.linalg.norm(dx)
		F_eval = F(x)
		grad = F_eval.T @ f_eval_new
		

		#########################################################################
		# Printing section 
		if s[-1] == 0:
			cond = np.inf
		else:	
			cond = s[0] / s[-1]
		
		if verbose >= 1:
			normgrad = np.linalg.norm(grad)
			print(
				'    %3d  |  %1.4e  |  %8.2e  |  %8.2e  |  %8.2e  |  %8.2e' % (
				it, normf, normdx, cond, alpha, normgrad))
		# Termination conditions
		if normgrad < tol:
			if verbose: print("norm gradient %1.3e less than tolerance %1.3e" % 
				(normgrad, tol))
			break
		if normdx < tol_normdx:
			if verbose: print("norm dx %1.3e less than tolerance %1.3e" % 
				(normdx, tol_normdx))
			break
		if residual_increased:
			if verbose: print("residual increased during line search from %1.5e to %1.5e" % 
				(np.linalg.norm(f_eval), np.linalg.norm(f_eval_new)))
			break

	if normgrad <= tol:
		info = 0
		if verbose >= 1:
			print('Gauss-Newton converged successfully!')
	elif normdx <= tol_normdx:
		info = 1
		if verbose >= 1:
			print ('Gauss-Newton did not converge: ||dx|| < tol')
	elif it == maxiter - 1:
		info = 2
		if verbose >= 1:
			print ('Gauss-Newton did not converge: max iterations reached')
	elif np.linalg.norm(f_eval_new) >= np.linalg.norm(f_eval):
		info = 3
		if verbose >= 1:
			print ('No progress made during line search')
	else:
		raise Exception('Stopping criteria not determined!')

	return x, info


