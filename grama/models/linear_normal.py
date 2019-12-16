__all__ = ["make_linear_normal"]

import numpy as np
from .. import core
from .. import compositions as cp

def limit_state(x):
    x1, x2 = x

    return 1 - x1 - x2

def make_linear_normal():
    md = core.Model("Linear-Normal Reliability Problem") >> \
         cp.cp_function(
             fun=limit_state,
             var=2,
             out=["g_linear"],
             name="limit state"
         ) >> \
         cp.cp_marginals(
             x0={"dist": "norm", "loc": 0, "scale": 1, "sign":+1},
             x1={"dist": "norm", "loc": 0, "scale": 1, "sign":+1}
         )

    return md
