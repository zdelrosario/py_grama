__all__ = ["make_linear_normal"]

import numpy as np
from .. import core
from scipy.stats import norm

def limit_state(x):
    x1, x2 = x

    return 1 - x1 - x2

class make_linear_normal(core.model):
    def __init__(self):
        super().__init__(
            name="Linear-Normal",
            functions=[
                core.function(
                    limit_state,
                    ["x1", "x2"],
                    ["g_linear"],
                    "limit state"
                )
            ],
            domain=core.domain(
                bounds = {
                    "x1": [-np.Inf, +np.Inf],
                    "x2": [-np.Inf, +np.Inf]
                }
            ),
            density=core.density(
                marginals=[
                    core.marginal_named(
                        "x1",
                        sign=+1,
                        d_name="norm",
                        d_param={"loc": 0, "scale": 1}
                    ),
                    core.marginal_named(
                        "x2",
                        sign=+1,
                        d_name="norm",
                        d_param={"loc": 0, "scale": 1}
                    )
                ]
            )
        )
