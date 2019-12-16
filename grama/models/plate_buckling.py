__all__ = ["make_plate_buckle"]

import numpy as np

from .. import core
from .. import compositions as cp

LOAD      = 0.00128 # Applied load (kips)

## Distributions fitted from Stang et al. data
MU_LOG_E  = 10301. # Young's modulus log-mean (kips / in^2)
SIG_LOG_E = 0.0256 # Young's modulus log-std  (kips / in^2)
A_NU      = 0.3100 # Poisson's ratio upper-bound
B_NU      = 0.3310 # Poisson's ratio lower-bound
# MU_LOG_L  =-6.6610 # Applied load log-mean (log kips)
MU_LOG_L  = LOAD # Applied load log-mean (log kips)
SIG_LOG_L = 0.01   # Applied load log-std (log kips)
RHO       = 0.6677 # Correlation

## Reference geometry
THICKNESS = 0.25 # Plate thickness (in)
HEIGHT    = 12. # Plate height    (in)

def function_buckle_state(x):
    t, h, E, nu, L = x
    return np.pi * E / 12 / (1 - nu**2) * (t / h)**2 - L

def make_plate_buckle():
    md = core.Model("Plate Buckling") >> \
         cp.cp_function(
             fun=function_buckle_state,
             var=["t", "h", "E", "nu", "L"],
             out=["g_buckle"],
             name="limit state"
         ) >> \
         cp.cp_bounds(t=(0, 2 * THICKNESS), h=(6, 18)) >> \
         cp.cp_marginals(
             E={"dist": "lognorm", "loc": 1, "s": SIG_LOG_E, "scale": MU_LOG_E},
             nu={"dist": "uniform", "loc": A_NU, "scale": B_NU - A_NU},
             L={
                 "dist": "lognorm",
                 "loc": 1, "s": SIG_LOG_L, "scale": MU_LOG_L,
                 "sign": +1
             }
         )

    return md
