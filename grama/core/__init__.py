from functools import partial
from .core import model_, eval_df

## Helper functions
##################################################
# Infix to help define pipe
class Infix(object):
    def __init__(self, func):
        self.func = func
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return Infix(partial(self.func, other))
    def __call__(self, v1, v2):
        return self.func(v1, v2)

# Pipe function
@Infix
def pi(x, f):
    """Infix pipe operator.
    """
    return f(x)
