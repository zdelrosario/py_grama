from __future__ import division

import numpy as np

from .random import RandomDomain
from .box import BoxDomain

class UniformDomain(BoxDomain, RandomDomain):
	r""" A randomized version of a BoxDomain with a uniform measure on the space.
	"""
	
	def _pdf(self, X):
		area = np.prod([ (ub - lb) for lb, ub in zip(self.lb, self.ub)])
		return self.isinside(X).astype(np.float)/area
