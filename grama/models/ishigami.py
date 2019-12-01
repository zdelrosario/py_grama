__all__ = ["make_ishigami"]

import numpy as np

from collections import OrderedDict as od
from .. import core

def fcn(x):
    a, b, x1, x2, x3 = x
    return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

class make_ishigami(core.model):
    def __init__(self):
        super().__init__(
            name="Ishigami",
            function=lambda x: fcn(x),
            outputs=["f"],
            domain=core.domain(
                bounds=od([
                    ("a",   [   6.0,    8.0]),
                    ("b",   [   0.0,    0.2]),
                    ("x1",  [-np.pi, +np.pi]),
                    ("x2",  [-np.pi, +np.pi]),
                    ("x3",  [-np.pi, +np.pi])
                ])
            ),
            density=core.density(
                marginals=[
                    core.marginal_named(
                        "x1",
                        d_name="uniform",
                        d_param={"loc": -np.pi, "scale": 2*np.pi}
                    ),
                    core.marginal_named(
                        "x2",
                        d_name="uniform",
                        d_param={"loc": -np.pi, "scale": 2*np.pi}
                    ),
                    core.marginal_named(
                        "x3",
                        d_name="uniform",
                        d_param={"loc": -np.pi, "scale": 2*np.pi}
                    )
                ]
            )
        )
