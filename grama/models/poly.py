import numpy as np

from .. import core
from scipy.stats import uniform

def function_poly(x):
    x1, x2, x3 = x

    return np.array([x1, x2**2, x3**3])

class model_poly(core.model_):
    def __init__(self):
        super().__init__(
            name     = "Polynomials",
            function = lambda x: function_poly(x),
            outputs  = ["p0", "p1", "p2"],
            domain   = core.domain_(
                hypercube = True,
                inputs    = ["x0", "x1", "x2"],
                bounds    = {
                    "x0":  [-1, +1],
                    "x1":  [-1, +1],
                    "x2":  [-1, +1]
                }
            ),
            density  = core.density_(
                pdf = lambda x: \
                uniform.pdf(x[0], loc = -1, scale = 2) * \
                uniform.pdf(x[1], loc = -1, scale = 2) * \
                uniform.pdf(x[2], loc = -1, scale = 2),
                pdf_factors = ["uniform", "uniform", "uniform"],
                pdf_param   = [
                    {"loc": -1, "scale": 2},
                    {"loc": -1, "scale": 2},
                    {"loc": -1, "scale": 2},
                ],
                pdf_qt_sign = [0, 0, 0]
            )
        )
