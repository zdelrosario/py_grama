""" Local linear models for use in other functions
"""
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import xlogy
from scipy.spatial.distance import cdist
import scipy.linalg

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

__all__ = ['perplexity_bandwidth', 'local_linear_grads', 'local_linear']

@lru_cache(maxsize = int(10))
def _compute_p1(M, perplexity):
	r""" The constant appearing in VC13, eq. 9
	"""
	#res = root_scalar(lambda x: np.log(min(np.sqrt(2*M), perplexity)) - 2*(1 - x)*np.log(M/(2*(1 - x))), bracket = [3./4,1 - 1e-14],)
	#p1 = res.root
	# Instead we solve in terms of x = 2*(1-p1)
	# x * log(N/x) = x * log(N) - x*log(x)
	# and xlogy(x, x) = x * log(x)
	fun = lambda x: x * np.log(M) - xlogy(x, x) - np.log(np.min([np.sqrt(2*M), perplexity]))

	try:
		res = root_scalar(fun, bracket = [0, 0.5])
		x = res.root
	except ValueError:
		if np.isclose(fun(0.5), 0):
			x = 0.5
		elif np.isclose(fun(0), 0):
			x = 0
		else:
			raise ValueError
 
	p1 = 1. - x/2.
	return p1


def log_entropy(beta, d):
	p = np.exp(-beta*d)
	sum_p = np.sum(p)
	# Shannon entropy H = np.sum(-p*np.log2(p)) 
	# More stable formula 
	return beta*np.sum(p*d/sum_p) + np.log(sum_p)


def perplexity_bandwidth(d, perplexity):
	r"""Compute the bandwidth such that the exponential kernel has the desired perplexity

	Parameters
	----------
	d: array-like (M,)
		List of squared Euclidean distances
	perplexity: float, positive
		Target entropy of 

	Returns
	-------
	bandwidth: float
		Bandwidth of Gaussian kernel (includes 1/2 factor)
	"""
	M = len(d)
	# TODO: Is perplexity necessarily in this interval
	#perplexity = min(M, perplexity)

	p1 = _compute_p1(M, perplexity)
	# Compute upper and lower bounds of beta from [VC13, eq. (7) (8)]
	# These are constants appearing the bounds
	dM = np.max(d)
	d1 = np.min(d[d>0])
	delta2 = d - d1
	delta2 = np.min(delta2[delta2>0])
	deltaM = dM - d1

	# lower bound VC13 (7)
	beta1 = max(M*np.log(M/perplexity)/((M-1)*deltaM), np.sqrt(np.log(M/perplexity)/(dM**4 - d1**4)))
	# upper bound VC13 (8)
	beta2 = 1/delta2*np.log(p1/(1-p1)*(M - 1))
	
	log_perplexity = np.log(perplexity)

	# TODO: Ideally we would use (root) Newton to find the optimal bandwidth
	# however Scipy doesn't support bounding intervals for its newton implementation.

	# TODO: There are some fancy initialization strategies that can be used
	# based on caching previous values; we ignore this here

	#print("beta1", beta1, log_entropy(beta1, d), "\nbeta2", beta2, log_entropy(beta2,d))


	# Compute bandwidth beta
	try:
		res = root_scalar(lambda beta: log_entropy(beta, d) - log_perplexity,
			bracket = [beta1, beta2],
			method = 'brenth',
			rtol = 1e-10)
		beta = res.root
	except ValueError as e:
		f1 = log_entropy(beta1, d) - log_perplexity
		f2 = log_entropy(beta2, d) - log_perplexity
		print("beta", beta1, "f", f1)
		print("beta", beta2, "f", f2)
		raise e
	return beta


def local_linear(X, fX, perplexity = None, bandwidth = None, Xt = None):
	r""" Construct local linear models at specified points 

	In several dimension reduction settings we want to estimate gradients using only samples.
	If we had freedom to place these samples anywhere we wanted, we would use a finite difference
	approach.  As this is often not the case, we need some way to estimate gradients given a 
	fixed and arbitrary set of data.

	Local linear models provide one approach for estimating the gradient. 
	This approach constructs a local linear model centered around each :math:`\mathbf{x}_t`
	with weights depending on the distance between points

	.. math::
		\min_{a_0\in \mathbb{R}, \mathbf{a}\in \mathbb{R}^m}
			\sum_{i=1}^M [(a_0 + \mathbf{a}^\top \mathbf{x}_i) - f(\mathbf{x}_i)]^2 e^{-\beta_t \| \mathbf{x}_i - \mathbf{x}_t\|_2^2}.	

	The choice of :math:`\beta_t` is critical. Here we provide two main options.  
	By default, we choose :math:`\beta_t` for each :math:`\mathbf{x}_t` 
	such that the perplexity corresponds to :math:`m+1`; other values of perplexity are avalible setting :code:`perplexity`.
	The other option is to specify the bandwidth :math:`\beta` explicitly.
	

	*Note* The cost of this method scales quadratically in the dimension of input space.	

	Parameters
	----------
	X: array-like (M, m)
		Places where the function is evaluated
	fX: array-like (M,)
		Value of the function at those locations
	perplexity: None or float
		If None, defaults to m+1.
	bandwidth: None, 'xia' or positive float
		If specified, set the global bandwidth to the specified float.
		If 'xia', use the bandwidth selection heuristic of Xia mentioned
		in [Li18]_. 

	Returns
	-------
	A: np.array (M, m+1)
		Matrix of coefficients of linear model; 
		A[:,0] is the constant term and A[:,1:m+1] is the linear coefficients.
	"""

	M, m = X.shape
	fX = fX.flatten()
	assert len(fX) == M, "Number of function evaluations does not match number of samples"

	if Xt is None:
		Xt = X

	if perplexity is None and bandwidth is None:
		perplexity = min(m+1, M)
	if perplexity is not None:
		bandwidth = None
		assert perplexity >= 2 and perplexity < M, "Perplexity must be in the interval [2,M)"
	elif bandwidth is not None:
		perplexity = None
		if bandwidth == 'xia':
			# Bandwidth from Xia 2007, [Li18, eq. 11.5] 
			bandwidth = 2.34*M**(-1./(max(M, 3) +6))


	# Storage for the gradients
	grads = np.zeros((len(Xt), m))
	Y = np.hstack([np.ones((M, 1)), X])
	
	# Coefficients in linear models
	A = np.zeros((M, m+1))
	for i, xi in enumerate(Xt):
		d = cdist(X, xi.reshape(1,-1), 'sqeuclidean').flatten()
		if perplexity:
			beta = perplexity_bandwidth(d, perplexity)
		else:
			beta = 0.5*bandwidth
	
		try:
			# Weights associated with each point
			sqrt_weights = np.exp(-0.5*beta*d).reshape(-1,1)
			#a, _, _, _ = np.linalg.lstsq(sqrt_weights*Y, sqrt_weights*fX.reshape(-1,1), rcond = None)
			a, _, _, _ = scipy.linalg.lstsq(sqrt_weights*Y, sqrt_weights*fX.reshape(-1,1), overwrite_a = True, overwrite_b = True)
			a = a.flatten()
		except np.linalg.LinAlgError:
			a = np.zeros(m+1)
		A[i,:] = a
	return A

def local_linear_grads(X, fX, perplexity = None, bandwidth = None, Xt = None):
	r""" Estimates the gradient from a local linear model
	"""
	A = local_linear(X, fX, perplexity = perplexity, bandwidth = bandwidth, Xt = Xt)
	return A[:,1:]
