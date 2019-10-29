import numpy as np
from .. import core
from scipy.stats import norm

def limit_state(x):
    x1, x2 = x

    return 1 - x1 - x2

class model_linear_normal(core.model_):
    def __init__(self, rho = 0):
        super().__init__(
            name = "Linear-Normal",
            function = limit_state,
            outputs = ["g_linear"],
            domain = core.domain_(
                hypercube = True,
                inputs = ["x1", "x2"],
                bounds = {
                    "x1": [-np.Inf, +np.Inf],
                    "x2": [-np.Inf, +np.Inf]
                }
            ),
            density = core.density_(
                pdf = lambda x: norm.pdf(x[0]) * norm.pdf(x[1]),
                pdf_factors = ["norm", "norm"],
                pdf_param = [
                    {"loc": 0, "scale": 1},
                    {"loc": 0, "scale": 1}
                ],
                pdf_qt_sign = [+1, +1]
            )
        )
