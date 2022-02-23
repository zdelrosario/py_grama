from __future__ import division

import numpy as np
import itertools

from ..misc import merge
from .euclidean import EuclideanDomain
from .domain import Domain
from .domain import TOL, DEFAULT_CVXPY_KWARGS

class TensorProductDomain(EuclideanDomain):
	r""" A class describing a tensor product of a multiple domains


	Parameters
	----------
	domains: list of domains
		Domains to combine into a single domain
	**kwargs
		Additional keyword arguments to pass to CVXPY
	"""
	def __init__(self, domains = None, **kwargs):
		self._domains = []
		if domains == None:
			domains = []
	
		for domain in domains:
			assert isinstance(domain, Domain), "Input must be list of domains"
			if isinstance(domain, TensorProductDomain):
				# If one of the sub-domains is a tensor product domain,
				# flatten it to limit recursion
				self._domains.extend(domain.domains)
			else:
				self._domains.append(domain)
		self.kwargs = merge(DEFAULT_CVXPY_KWARGS, kwargs)

	def __str__(self):
		return "<TensorProductDomain on R^%d of %d domains>" % (len(self), len(self.domains))
		
	@property
	def names(self):
		return list(itertools.chain(*[dom.names for dom in self.domains]))

	def _is_linquad_domain(self):
		return all([dom.is_linquad_domain for dom in self.domains])
	
	def _is_linineq_domain(self):
		return all([dom.is_linineq_domain for dom in self.domains])
		
	def _is_box_domain(self):
		return all([dom.is_box_domain for dom in self.domains])

	@property
	def domains(self):
		return self._domains

	@property
	def _slices(self):	
		start, stop = 0,0
		for dom in self.domains:
			stop += len(dom)
			yield(slice(start, stop))
			start += len(dom) 	

	def _sample(self, draw = 1):
		X = []
		for dom in self.domains:
			X.append(dom.sample(draw = draw))
		return np.hstack(X)

	def _isinside(self, X, tol = TOL):
		inside = np.ones(X.shape[0], dtype = np.bool)
		for dom, I in zip(self.domains, self._slices):
			#print(dom, I, dom.isinside(X[:,I]))
			inside = inside & dom.isinside(X[:,I], tol = tol)
		return inside

	def _extent(self, x, p):
		alpha = [dom._extent(x[I], p[I]) for dom, I in zip(self.domains, self._slices)]
		return min(alpha)


	def __len__(self):
		return sum([len(dom) for dom in self.domains])


	################################################################################		
	# Normalization related functions
	################################################################################		
	
	def _normalize(self, X):
		return np.hstack([dom.normalize(X[:,I]) for dom, I in zip(self.domains, self._slices)])

	def _unnormalize(self, X_norm):
		return np.hstack([dom.unnormalize(X_norm[:,I]) for dom, I in zip(self.domains, self._slices)])

	def _unnormalize_der(self):
		return np.diag(np.hstack([np.diag(dom._unnormalize_der()) for dom in self.domains]))

	def _normalize_der(self):
		return np.diag(np.hstack([np.diag(dom._normalize_der()) for dom in self.domains]))

	def _normalized_domain(self, **kwargs):
		domains_norm = [dom.normalized_domain(**kwargs) for dom in self.domains]
		return TensorProductDomain(domains = domains_norm)

	
	################################################################################		
	# Convex solver problems
	################################################################################		
	
	def _build_constraints_norm(self, x_norm):
		constraints = []
		for dom, I in zip(self.domains, self._slices):
			constraints.extend(dom._build_constraints_norm(x_norm[I]))
		return constraints 


	def _build_constraints(self, x):
		constraints = []
		for dom, I in zip(self.domains, self._slices):
			constraints.extend(dom._build_constraints(x[I]))
		return constraints 

	################################################################################		
	# Properties resembling LinQuad Domains 
	################################################################################		

	@property
	def lb(self):
		return np.concatenate([dom.lb for dom in self.domains])

	@property
	def ub(self):
		return np.concatenate([dom.ub for dom in self.domains])

	@property
	def A(self):
		A = []
		for dom, I in zip(self.domains, self._slices):
			A_tmp = np.zeros((dom.A.shape[0] ,len(self)))
			A_tmp[:,I] = dom.A
			A.append(A_tmp)
		return np.vstack(A)

	@property
	def b(self):
		return np.concatenate([dom.b for dom in self.domains])

	@property
	def A_eq(self):
		A_eq = []
		for dom, I in zip(self.domains, self._slices):
			A_tmp = np.zeros((dom.A_eq.shape[0] ,len(self)))
			A_tmp[:,I] = dom.A_eq
			A_eq.append(A_tmp)
		return np.vstack(A_eq)
	
	@property
	def b_eq(self):
		return np.concatenate([dom.b_eq for dom in self.domains])

	def add_constraints(self, *args, **kwargs):
		if self.is_linquad_domain:
			from .linquad import LinQuadDomain
			return LinQuadDomain.add_constraints(self, *args, **kwargs)
		else:
			raise NotImplementedError

	def _init_lb(self, *args, **kwargs):
		from .linquad import LinQuadDomain
		return LinQuadDomain._init_lb(self, *args, **kwargs)
	
	def _init_ub(self, *args, **kwargs):
		from .linquad import LinQuadDomain
		return LinQuadDomain._init_ub(self, *args, **kwargs)
	
	def _init_ineq(self, *args, **kwargs):
		from .linquad import LinQuadDomain
		return LinQuadDomain._init_ineq(self, *args, **kwargs)
	
	def _init_eq(self, *args, **kwargs):
		from .linquad import LinQuadDomain
		return LinQuadDomain._init_eq(self, *args, **kwargs)

	def _init_quad(self, *args, **kwargs):
		from .linquad import LinQuadDomain
		return LinQuadDomain._init_quad(self, *args, **kwargs)

