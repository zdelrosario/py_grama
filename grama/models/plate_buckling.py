import numpy as np

from .. import core
from scipy.stats import lognorm, uniform

LOAD      = 0.00128 # Applied load (kips)

## Distributions fitted from Stang et al. data
MU_LOG_E  = 9.2429 # Young's modulus log-mean (log 10^3 kips / in^2)
SIG_LOG_E = 0.0256 # Young's modulus log-std  (log 10^3 kips / in^2)
A_NU      = 0.3100 # Poisson's ratio upper-bound
B_NU      = 0.3310 # Poisson's ratio lower-bound
RHO       = 0.6677 # Correlation

## Reference geometry
THICKNESS = 0.25 # Plate thickness (in)
HEIGHT    = 12. # Plate height    (in)

def function_buckle_state(x, t = THICKNESS, h = HEIGHT):
    E, nu = x
    return np.pi * E / 12 / (1 - nu**2) * (t / h)**2 - LOAD

class model_plate_buckle(core.model_):
    def __init__(self, t = THICKNESS, h = HEIGHT):
        super().__init__(
            name     = "Plate Buckling",
            function = lambda x: function_buckle_state(x, t = t, h = h),
            outputs  = ["g_buckle"],
            domain   = core.domain_(
                hypercube = True,
                inputs    = ["E", "nu"],
                bounds    = {
                    "E":  [0, +np.Inf],
                    "nu": [A_NU, B_NU]
                }
            ),
            density  = core.density_(
                pdf = lambda X: \
                lognorm.pdf(X[2], s = SIG_LOG_E, loc = MU_LOG_E, scale = 1) * \
                uniform.pdf(X[3], loc = A_NU, scale = B_NU - A_NU),
                pdf_factors = ["lognorm", "uniform"],
                pdf_param   = [
                    {"s": SIG_LOG_E, "loc": MU_LOG_E, "scale": 1},
                    {"loc": A_NU, "scale": B_NU - A_NU},
                ],
                # pdf_qt_flip = [0, 1]
                pdf_qt_sign = [-1, +1]
            )
        )
