__all__ = ["make_cantilever_beam"]

import numpy as np
from .. import core
from collections import OrderedDict as od
from numpy import sqrt, array, Inf
from scipy.stats import norm

LENGTH = 100
D_MAX  = 2.2535

MU_H   = 500.
MU_V   = 1000.
MU_E   = 2.9e7
MU_Y   = 40000.

TAU_H  = 100.
TAU_V  = 100.
TAU_E  = 1.45e6
TAU_Y  = 2000.

def function_beam(x):
    w, t, H, V, E, Y = x

    return array([
        w * t,
        Y - 600 * V / w / t**2 - 600 * H / w**2 / t,
        D_MAX - np.float64(4) * LENGTH**3 / E / w / t * sqrt(
            V**2 / t**4 + H**2 / w**4
        )
    ])

class make_cantilever_beam(core.model):
    def __init__(self):
        bounds = od()
        bounds["w"] = [2, 4]
        bounds["t"] = [2, 4]
        bounds["H"] = [-Inf, +Inf]
        bounds["V"] = [-Inf, +Inf]
        bounds["E"] = [-Inf, +Inf]
        bounds["Y"] = [-Inf, +Inf]

        super().__init__(
            name="Cantilever Beam",
            function=lambda x: function_beam(x),
            outputs=["c_area", "g_stress", "g_displacement"],
            domain=core.domain(bounds=bounds),
            density=core.density(
                marginals=[
                    core.marginal_named(
                        "H",
                        sign=+1,
                        d_name="norm",
                        d_param={"loc": MU_H, "scale": TAU_H}
                    ),
                    core.marginal_named(
                        "V",
                        sign=+1,
                        d_name="norm",
                        d_param={"loc": MU_V, "scale": TAU_V}
                    ),
                    core.marginal_named(
                        "E",
                        sign=0,
                        d_name="norm",
                        d_param={"loc": MU_E, "scale": TAU_E}
                    ),
                    core.marginal_named(
                        "Y",
                        sign=-1,
                        d_name="norm",
                        d_param={"loc": MU_Y, "scale": TAU_Y}
                    )
                ]
            )
        )
