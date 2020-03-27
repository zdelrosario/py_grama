__all__ = [
    "sin",
    "cos",
    "log",
    "exp",
    "sqrt",
    "pow",
    "as_int",
    "as_float",
    "as_str",
]

from grama import make_symbolic

from numpy import sin as npsin
from numpy import cos as npcos
from numpy import log as nplog
from numpy import exp as npexp
from numpy import sqrt as npsqrt
from numpy import power as nppower

# --------------------------------------------------
# Mutation helpers
# --------------------------------------------------
# Numeric
@make_symbolic
def sin(x):
    return npsin(x)


@make_symbolic
def cos(x):
    return npcos(x)


@make_symbolic
def log(x):
    return nplog(x)


@make_symbolic
def exp(x):
    return npexp(x)


@make_symbolic
def sqrt(x):
    return npsqrt(x)


@make_symbolic
def pow(x, p):
    return nppower(x, p)


# Casting
@make_symbolic
def as_int(x):
    return x.astype(int)


@make_symbolic
def as_float(x):
    return x.astype(float)


@make_symbolic
def as_str(x):
    return x.astype(str)
