import numpy as np
import scipy.linalg
from .misc import check_sample_inputs
from .local_linear import local_linear_grads  

def initialize_subspace(X = None, fX = None, grads = None, n_grads = 100):
	r""" Construct an initial estimate of the desired subspace 
	"""


	X, fX, grads = check_sample_inputs(X, fX, grads)
	ngrads = max(n_grads, X.shape[1])
	# If we don't have enough grads and we have enough samples to estimate gradients
	if len(grads) < n_grads and X.shape[0] >= X.shape[1]+1:
		# Pick a random subset to estimate the gradient at
		I = np.random.permutation(X.shape[0])[:n_grads - len(grads)]
		grad_est = local_linear_grads(X, fX, Xt = X[I])
		all_grads = np.vstack([grads, grad_est])
	else:
		all_grads = grads

	# Compute SVD
	U, s, VH = scipy.linalg.svd(all_grads.T, full_matrices = False, compute_uv = True)
	return U	
	
