from __future__ import print_function
import numpy as np
import textwrap
import inspect

#from .domains import Domain

from .misc import merge


__all__ = ['BaseFunction']

class BaseFunction(object):
	r""" Abstract base class for functions

	"""
	def eval(self, X, **kwargs):
		return self.__call__(X, return_grad = False)

	def grad(self, X):
		return self.__call__(X, return_grad = True)[1]

	def hessian(self, X):
		raise NotImplementedError

	def __call__(self, X, return_grad = False, **kwargs):
		if return_grad:
			return self.eval(X, **kwargs), self.grad(X)
		else:
			return self.eval(X, **kwargs)

	def predict(self, X):
		r""" Alias of __call__ to match scikit learn API
		"""
		return self.__call__(X)
