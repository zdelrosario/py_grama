__all__ = ["make_poly"]

import numpy as np

from collections import OrderedDict as od
from .. import core
from scipy.stats import uniform

class make_poly(core.model):
    def __init__(self):
        super().__init__(
            name="Polynomials",
            functions=[
                core.function(
                    lambda x: x,
                    ["x0"],
                    ["p0"],
                    "linear"
                ),
                core.function(
                    lambda x: x,
                    ["x1"],
                    ["p1"],
                    "quadratic"
                ),
                core.function(
                    lambda x: x,
                    ["x2"],
                    ["p2"],
                    "cubic"
                )
            ],
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
