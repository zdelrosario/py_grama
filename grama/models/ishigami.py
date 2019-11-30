import numpy as np

from .. import core

def fcn(x, a, b):
    x1, x2, x3 = x
    return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

class model_ishigami(core.model_):
    def __init__(self, a=7.0, b=0.1):
        super().__init__(
            name     = "Ishigami",
            function = lambda x: fcn(x, a, b),
            outputs  = ["f"],
            domain   = core.domain_(
                hypercube = True,
                inputs    = ["x1", "x2", "x3"],
                bounds    = {
                    "x1":  [-np.pi, +np.pi],
                    "x2":  [-np.pi, +np.pi],
                    "x3":  [-np.pi, +np.pi]
                }
            ),
            density  = core.density_(
                pdf = lambda x: \
                uniform.pdf(x[0], loc=-np.pi, scale=2*np.pi) * \
                uniform.pdf(x[1], loc=-np.pi, scale=2*np.pi) * \
                uniform.pdf(x[2], loc=-np.pi, scale=2*np.pi),
                pdf_factors = ["uniform", "uniform", "uniform"],
                pdf_param   = [
                    {"loc": -np.pi, "scale": 2*np.pi},
                    {"loc": -np.pi, "scale": 2*np.pi},
                    {"loc": -np.pi, "scale": 2*np.pi},
                ],
                pdf_qt_sign = [0, 0, 0]
            )
        )
