
class EmptyDomainException(Exception):
	r""" Raised when trying to call an operation that does not work on an empty domain
	"""
	pass

class UnboundedDomainException(Exception):
	r""" Raised when trying to preform an operation on a domain that is ill-posed on an unbounded domain
	"""
	pass

class SolverError(Exception):
	r""" A problem with the solver
	"""
	pass	

class UnderdeterminedException(Exception):
	pass
