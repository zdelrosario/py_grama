__all__ = ["make_poly"]

import numpy as np

from .. import core
from scipy.stats import uniform

class make_poly(core.Model):
    def __init__(self):
        super().__init__(
            name="Polynomials",
            functions=[
                core.Function(
                    lambda x: x,
                    ["x0"],
                    ["p0"],
                    "linear"
                ),
                core.Function(
                    lambda x: x,
                    ["x1"],
                    ["p1"],
                    "quadratic"
                ),
                core.Function(
                    lambda x: x,
                    ["x2"],
                    ["p2"],
                    "cubic"
                )
            ],
            domain=core.Domain(
                bounds    = dict([
                    ("x0",  [-1, +1]),
                    ("x1",  [-1, +1]),
                    ("x2",  [-1, +1])
                ])
            ),
            density=core.Density(
                marginals=[
                    core.MarginalNamed(
                        "x0",
                        d_name="uniform",
                        d_param={"loc": -1, "scale": 2}
                    ),
                    core.MarginalNamed(
                        "x1",
                        d_name="uniform",
                        d_param={"loc": -1, "scale": 2}
                    ),
                    core.MarginalNamed(
                        "x2",
                        d_name="uniform",
                        d_param={"loc": -1, "scale": 2}
                    )
                ]
            )
        )
