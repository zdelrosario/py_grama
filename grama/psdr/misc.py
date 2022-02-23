from __future__ import print_function, division
""" misc. utilities and defintions
"""
import numpy as np

def merge(x, y):
	z = x.copy()
	z.update(y)
	return z


def check_sample_inputs(X, fX, grads):
	if X is not None and fX is not None:
		X = np.array(X)
		fX = np.array(fX).flatten()
		assert len(X) == len(fX), "Number of samples doesn't match number of evaluations"
	else:
		X = None
		fX = None

	if grads is not None:
		grads = np.array(grads)
		if X is not None:
			assert X.shape[1] == grads.shape[1], "Dimensions of gradients doesn't match dimension of samples"

	if X is None:
		X = np.zeros((0, grads.shape[1]))
		fX = np.zeros((0,))
	if grads is None:
		grads = np.zeros((0, X.shape[1]))
	return X, fX, grads	
