__all__ = ["make_poly"]

import numpy as np

from .. import core
from .. import compositions as cp

def make_poly():
    md = core.Model("Polynomials") >> \
         cp.cp_function(fun=lambda x: x, var=1, out=1, name="linear") >> \
         cp.cp_function(fun=lambda x: x**2, var=1, out=1, name="quadratic") >> \
         cp.cp_function(fun=lambda x: x**3, var=1, out=1, name="cubic") >> \
         cp.cp_marginals(
             x0={"dist": "uniform", "loc": -1, "scale": 2},
             x1={"dist": "uniform", "loc": -1, "scale": 2},
             x2={"dist": "uniform", "loc": -1, "scale": 2}
         )

    return md
