__all__ = ["make_poly"]

import numpy as np

from collections import OrderedDict as od
from .. import core
from scipy.stats import uniform

def function_poly(x):
    x1, x2, x3 = x
    return np.array([x1, x2**2, x3**3])

class make_poly(core.model):
    def __init__(self):
        super().__init__(
            name="Polynomials",
            function=lambda x: function_poly(x),
            outputs=["p0", "p1", "p2"],
            domain=core.domain(
                bounds    = od([
                    ("x0",  [-1, +1]),
                    ("x1",  [-1, +1]),
                    ("x2",  [-1, +1])
                ])
            ),
            density=core.density(
                marginals=[
                    core.marginal_named(
                        "x0",
                        d_name="uniform",
                        d_param={"loc": -1, "scale": 2}
                    ),
                    core.marginal_named(
                        "x1",
                        d_name="uniform",
                        d_param={"loc": -1, "scale": 2}
                    ),
                    core.marginal_named(
                        "x2",
                        d_name="uniform",
                        d_param={"loc": -1, "scale": 2}
                    )
                ]
            )
        )
