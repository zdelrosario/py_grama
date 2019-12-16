__all__ = ["make_linear_normal"]

import numpy as np
from .. import core
from scipy.stats import norm

def limit_state(x):
    x1, x2 = x

    return 1 - x1 - x2

class make_linear_normal(core.Model):
    def __init__(self):
        super().__init__(
            name="Linear-Normal",
            functions=[
                core.Function(
                    limit_state,
                    ["x1", "x2"],
                    ["g_linear"],
                    "limit state"
                )
            ],
            domain=core.Domain(
                bounds = {
                    "x1": [-np.Inf, +np.Inf],
                    "x2": [-np.Inf, +np.Inf]
                }
            ),
            density=core.Density(
                marginals=[
                    core.MarginalNamed(
                        "x1",
                        sign=+1,
                        d_name="norm",
                        d_param={"loc": 0, "scale": 1}
                    ),
                    core.MarginalNamed(
                        "x2",
                        sign=+1,
                        d_name="norm",
                        d_param={"loc": 0, "scale": 1}
                    )
                ]
            )
        )
