r""" Tools for computing Voronoi vertices
"""

from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist, pdist
import scipy.linalg
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

from scipy.spatial import HalfspaceIntersection
from .cdist import cdist
from .geometry import unique_points

from scipy.spatial.qhull import QhullError



def voronoi_vertex_1d(domain, Xhat, L = None):
	r""" One dimensional voronoi vertices

	As Qhull only works on two or more dimensions, 
	this code handles the simple, one-dimensional case
	"""
	assert len(domain) == 1, "Domain must be one-dimensional"
	
	lb = domain.corner(np.array([-1]))
	ub = domain.corner(np.array([1]))
	
	X = np.sort(np.hstack([Xhat.flatten(), ub, lb]))
	# mid point between X
	V = 0.5*(X[1:] + X[:-1])
	# add boundaries
	V = np.hstack([lb, V, ub])
	return V.reshape(-1,1)

def voronoi_vertex(domain, Xhat, L = None):
	r""" Construct the bounded Voronoi vertices on a domain 




	Note: 
	Pro17 claims that these bounded Voronoi vertices can be computed using WP89.  
	This approach may offer some speedup, but I trust Q-hull more than my own implementation.

	"""
	if len(domain) == 1:
		return voronoi_vertex_1d(domain, Xhat)
	
	assert np.all(domain.isinside(Xhat))


	if L is None:
		L = np.eye(len(domain))
		Linv = np.eye(len(domain))
	else:
		U, s, VT = np.linalg.svd(L)
		Linv = VT.T @ np.diag(1./s) @ U.T



	# Half spaces from the domain
	A, b = domain.A_aug, domain.b_aug

	if len(Xhat) == 1:
		halfspaces = np.hstack([A, -b.reshape(-1,1)])
		hs = HalfspaceIntersection(halfspaces, Xhat[0])
		return hs.intersections	

	Yhat = (L @ Xhat.T).T
	LV = []
	for k, xhat in enumerate(Xhat):
		Ak = Yhat - Yhat[k] 
		#Ak = np.vstack([L @ (Xhat[j] - xhat) for j in range(len(Xhat)) if j != k])
		center = 0.5 * (Yhat[k] + Yhat)
		#center = np.vstack([ 0.5*L @ (xhat + Xhat[j]) for j in range(len(Xhat)) if j != k]) 
		bk = np.sum(Ak * center, axis = 1)
		
		# If two Xhat are the same, we can end up with a situation 
		# where the constraint stops  so we filter these out
		I = np.argwhere(np.sum(Ak**2, axis = 1) > 0).flatten()
		Ak, bk = Ak[I], bk[I]
	
		halfspaces = np.hstack([np.vstack([A @ Linv , Ak  ]) , -np.hstack([b, bk]).reshape(-1,1)])

		try:
			hs = HalfspaceIntersection(halfspaces, xhat)
		except QhullError:
			# The point xhat isn't strictly inside the constraints, so we try with a point that is
			# Here we use the code from scipy documentation 
			norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),(halfspaces.shape[0], 1))
			cc = np.zeros((halfspaces.shape[1],))
			cc[-1] = -1
			AA = np.hstack((halfspaces[:, :-1], norm_vector))
			bb = -halfspaces[:, -1:]
			
			# Solve the linear program with CVXOPT as scipy's linprog reports ill-conditioning 
			sol = solvers.lp(matrix(cc), matrix(AA), matrix(bb))
			xhat2 = np.array(sol['x'][:-1]).flatten()
			
			try:
				hs = HalfspaceIntersection(halfspaces, xhat2)
			except QhullError as e:
				raise e


		# TODO: Do we want to keep indexing information, or only return the vertices 
		LV.append(np.copy(hs.intersections))

	LV = np.vstack(LV)
	V = (Linv @ LV.T).T
	I = unique_points(V)
	V = V[I]
	
	I = domain.isinside(V)
	V = V[I]
	return V

def voronoi_vertex_sample(domain, Xhat, X0, L = None, randomize = True):
	r""" Constructs a subset of the Voronoi vertices on a given domain 


	Given a domain :math:`\mathcal{D} \subset \mathbb{R}^m`, a set of points
	:math:`\lbrace \widehat{\mathbf{x}}_j \rbrace_{j=1}^M`, and a distance metric
	
	.. math::
		
		d(\mathbf{x}_1, \mathbf{x}_2) = \|\mathbf{L}(\mathbf{x}_1 - \mathbf{x}_2)\|_2

	the vertices of the bounded Voronoi diagram :math:`\mathbf{v}_i \in \mathcal{D}`
	are those points are those points that satisfy :math:`r` equidistance constraints
	in the metric :math:`d`

	.. math::

		\| \mathbf{L}(\mathbf{v}_i - \widehat{\mathbf{x}}_{\mathcal{I}_i[1]})\|_2 =
		\| \mathbf{L}(\mathbf{v}_i - \widehat{\mathbf{x}}_{\mathcal{I}_i[2]})\|_2 = \ldots = 
		\| \mathbf{L}(\mathbf{v}_i - \widehat{\mathbf{x}}_{\mathcal{I}_i[r]})\|_2 

	and :math:`m-r` constraints from the domain.
	
	
	The bounded Voronoi vertices include the vertices of the domain.
	The number of domain vertices can grow exponentially in dimension (e.g., consider the cube),
	but the asymptotic cost of finding these is :math:`\mathcal{O}(mn)` [wikiVEP]_ 
	per vertex where :math:`n` is the number of linear inequality constraints specifying the domain.
	Hence, the complexity of finding the Voronoi vertices grows exponentially in the 
	dimension of the domain.
	Thus we use an approach due Lindemann and Cheng [LC05]_ to find a subset of vertices
	by taking a starting point :math:`\mathbf{x}_0` and sequentially projecting it
	onto a series of hyperplanes such that after :math:`m` steps we satisfy :math:`m` constraints.	

	Parameters
	----------
	domain: Domain
		Domain on which to construct the vertices
	Xhat: array-like (M, m)
		M existing points on the domain which define the Voronoi diagram
	X0: array-like (N, m)
		Initial points to use to find vertices
	L: array-like (m, m), optional
		Weight on distance in 2-norm; defaults to the identity matrix	
	randomize: bool, optional (default: True)
		If true, when L is rank deficient add a random direction at each step
		such that we find points on the boundary of the domain satisfying m
		constraints.


	Returns
	-------
	X: np.ndarray(N, m)
		Points satisfying m constraints


	References
	----------
	.. [LC05] Iteratively Locating Voronoi Vertices for Dispersion Estimation
		Stephen R. Lindemann and Peng Cheng
		Proceedings of the 2005 Interational Conference on Robotics and Automation
	.. [wikiVEP] Vertex enumeration problem, Wikipedia.
		https://en.wikipedia.org/wiki/Vertex_enumeration_problem
	"""

	# Startup checks
	#assert isinstance(domain, Domain), "domain must be an instance of the Domain class"
	assert len(domain.Ls) == 0, "Currently this does not support domains with quadratic constraints"
	Xhat = np.atleast_2d(np.array(Xhat))
	X0 = np.atleast_2d(np.array(X0)).copy()

	m = len(domain)
	
	if L is None:
		L = np.eye(m)
	else:
		L = np.array(L)
		assert L.shape[1] == m, "Number of columns doesn't match dimension of the space"

	Lrank = np.linalg.matrix_rank(L)

	LTL = L.T.dot(L)

	# Linear inequality constraints for the domain (including lower bound/upper bound box constraints)
	A = domain.A_aug
	b = domain.b_aug

	# This algorithm terminates when each point has m active constraints as we can have no more improvement
	# hence we substract the number of equality constraints
	# we also subtract the rank-deficency of L because this results in zero-steps  	
	for k in range(m - domain.A_eq.shape[0]):
		if k >= Lrank and not randomize:
			break

		# Find the nearest neighbors
		# As we intend to do this in high dimensions, 
		# we don't use a tree-based distance approach
		# as these don't scale well
		D = cdist(L.dot(X0.T).T, L.dot(Xhat.T).T)
		I = np.argsort(D, axis = 1)

		# set the search direction to move away from the closest point
		h = X0 - Xhat[I[:,0]]
		h = LTL.dot(h.T).T
			 

		# which inequality constraints on the domain are active
		active = np.isclose(A.dot(X0.T).T , b)
		
		for i in range(X0.shape[0]):
			# For each point, project onto the feasible directions
			nullspace = [domain.A_eq.T]

			# active inequality constraints
			for j in range(A.shape[0]):
				if active[i,j] : nullspace += [A[j].reshape(-1,1)]

			# constraints from other points in the domain
			for j in range(1, min(k+1, len(Xhat))):
				if np.isclose(D[i,I[i,0]], D[i,I[i,j]]):	
					nullspace += [LTL.dot(Xhat[I[i,0]] - Xhat[I[i,j]]).reshape(-1,1)]
					#h[i] += (X0[i] - Xhat[I[i,j]])

			nullspace = np.hstack(nullspace)
			# If there are no active constraints, don't do anything
			if nullspace.shape[1]>0:
				Q, R = scipy.linalg.qr(nullspace, overwrite_a = True, mode = 'economic')
				h[i] -= Q.dot(Q.T.dot(h[i]))
				if np.isclose(np.linalg.norm(h[i]), 0) and randomize:
					# If L is low-rank we can have situations where h[i]	
					# as constructed above is in the nullspace of constraints
					# and hence is approximately zero.  When randomize=True
					# we choose a random search direction so we can still make 
					# progress towards satisfying m constraints.
					h[i] = np.random.randn(*h[i].shape)
					h[i] -= Q.dot(Q.T.dot(h[i]))
				#print("k=%d, constraints %d, norm %g" % (k, Q.shape[1], np.linalg.norm(h[i])))
		
		# Now we find the furthest we can step along this direction before either hitting a
		# (1) separating hyperplane separating x0 and points in Xhat or 
		# (2) the boundary of the domain

		alpha = np.inf*np.ones(X0.shape[0])

		# (1) Take a step until we run into a separating hyperplane
		for xhat in Xhat:
			# we setup a hyperplane (p - p0)^* n  = 0 
			# where p0 is a point on the hyperplane
			# separating the closest point Xhat[:,I[:,0]] and xhat
			p0 = 0.5*(L.dot( (xhat + Xhat[I[:,0]]).T).T)
			n = L.dot( (xhat - Xhat[I[:,0]]).T).T

			# Inner product of normal n with search direction h
			nh = np.sum(n*L.dot(h.T).T, axis = 1)

			# Inner product for the numerator (x0 - p0)^* n
			numerator = -np.sum( (L.dot(X0.T).T - p0)*n, axis = 1)
			
			alpha_c = np.inf*np.ones(alpha.shape)
			act = ~np.isclose(np.abs(nh), 0) 
			alpha_c[act] = numerator[act]/nh[act]
			# When xhat is the closest point, we get a divide by zero error
			# We cannot move backwards, so these are also set to infinity
			alpha_c[alpha_c < 0 ] = np.inf
			alpha = np.minimum(alpha, alpha_c)

		# (2) Find intersection with the domain
		# cf., Domain._extent_ineq 
		AX0 = A.dot(X0.T).T
		Ah = A.dot(h.T).T
		with np.errstate(divide = 'ignore', invalid = 'ignore'):
			alpha_c = (b - AX0)/Ah
		alpha_c[~np.isfinite(alpha_c)] = np.inf
		alpha_c[alpha_c <= 0] = np.inf
		# When the step size is near zero, the above formula is inaccurate
		# causing points to emerge outside of the domain. Hence, we zero the
		# step size for these points.
		alpha_c[ np.max(np.abs(h), axis = 1) < 1e-10] = 0. 
		alpha_c = np.min(alpha_c, axis = 1)
		alpha = np.minimum(alpha, alpha_c)

		# There are some cases above where both alpha and alpha_c will yield an infinite step
		# in this case we zero these just to be safe
		alpha[~np.isfinite(alpha)] = 0.

		# Now finally take the step
		X0 += alpha.reshape(-1,1)*h

	
	# Keep only those points that are inside the domain
	# (all points should be, but sometimes numerical issues push outside)
	#X0 = X0[domain.isinside(X0)]

	return X0
