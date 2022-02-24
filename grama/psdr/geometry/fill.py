from __future__ import print_function, division

import numpy as np
from .cdist import cdist

from .vertex import voronoi_vertex_sample, voronoi_vertex

def fill_distance(domain, Xhat, L = None):
	r""" Compute the exact fill distance by constructing the bounded Voronoi vertices


	Parameters
	----------
	domain: Domain
		Domain on which to compute the dispersion
	Xhat: array-like (?, m)
		Existing samples on the domain
	L: array-like (?, m) optional
		Matrix defining the distance metric on the domain
	
	Returns
	-------
	d: float
		Fill distance 
	"""

	V = voronoi_vertex(domain, Xhat, L = L)
	D = cdist(Xhat, V, L = L)
	return np.max(np.min(D, axis= 0))	

def fill_distance_estimate(domain, Xhat, L = None, Nsamp = int(1e3), X0 = None ):
	r""" Estimate the fill distance of the points Xhat in the domain

	The *fill distance* (Def. 1.4 of [Wen04]_) or *dispersion* [LC05]_
	is the furthest distance between any point :math:`\mathbf{x} \in \mathcal{D}` 
	and a set of points 
	:math:`\lbrace \widehat{\mathbf{x}}_j \rbrace_{j=1}^m \subset \mathcal{D}`:
		
	.. math::

		\sup_{\mathbf{x} \in \mathcal{D}} \min_{j=1,\ldots,M} \|\mathbf{L}(\mathbf{x} - \widehat{\mathbf{x}}_j)\|_2.

	Similar to :meth:`psdr.seq_maximin_sample`, this uses :meth:`psdr.voronoi_vertex_sample` to find 
	a subset of local maximizers and returns the best of these.

	Parameters
	----------
	domain: Domain
		Domain on which to compute the dispersion
	Xhat: array-like (?, m)
		Existing samples on the domain
	L: array-like (?, m) optional
		Matrix defining the distance metric on the domain
	Nsamp: int, default 1e4
		Number of samples to use for vertex sampling
	X0: array-like (?, m)
		Samples from the domain to use in :meth:`psdr.voronoi_vertex_sample`

	Returns
	-------
	d: float
		Fill distance lower bound

	References
	----------
	..  [Wen04] Scattered Data Approximation. Holger Wendland.
			Cambridge University Press, 2004.
			https://doi.org/10.1017/CBO9780511617539
	.. [LC05] Iteratively Locating Voronoi Vertices for Dispersion Estimation
		Stephen R. Lindemann and Peng Cheng
		Proceedings of the 2005 Interational Conference on Robotics and Automation
	"""
	if L is None:
		L = np.eye(len(domain))
	
	if X0 is None:
		from ..sample import initial_sample
		X0 = initial_sample(domain, L, Nsamp = Nsamp)

	# Since we only care about distance in L, we can terminate early if L is rank-deficient
	# and hence we turn off the randomization
	Xcan = voronoi_vertex_sample(domain, Xhat, X0, L = L, randomize = False)

	# Euclidean distance
	D = cdist(Xcan, Xhat, L)

	d = np.min(D, axis = 1)
	return float(np.max(d))

