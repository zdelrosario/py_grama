__all__ = ["make_ishigami"]

import numpy as np

from .. import core

def fcn(x):
    a, b, x1, x2, x3 = x
    return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

class make_ishigami(core.Model):
    def __init__(self):
        super().__init__(
            name="Ishigami",
            functions=[
                core.Function(
                    fcn,
                    ["a", "b", "x1", "x2", "x3"],
                    ["f"],
                    "ishigami function"
                )
            ],
            domain=core.Domain(bounds={"a":[6.0,8.0], "b": [0.0,0.2]}),
            density=core.Density(
                marginals=[
                    core.MarginalNamed(
                        "x1",
                        d_name="uniform",
                        d_param={"loc": -np.pi, "scale": 2*np.pi}
                    ),
                    core.MarginalNamed(
                        "x2",
                        d_name="uniform",
                        d_param={"loc": -np.pi, "scale": 2*np.pi}
                    ),
                    core.MarginalNamed(
                        "x3",
                        d_name="uniform",
                        d_param={"loc": -np.pi, "scale": 2*np.pi}
                    )
                ]
            )
        )
