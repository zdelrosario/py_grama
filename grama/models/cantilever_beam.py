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

def function_area(x):
    w, t = x
    return w * t

def function_stress(x):
    w, t, H, V, E, Y = x
    return Y - 600 * V / w / t**2 - 600 * H / w**2 / t

def function_displacement(x):
    w, t, H, V, E, Y = x
    return D_MAX - np.float64(4) * LENGTH**3 / E / w / t * sqrt(
        V**2 / t**4 + H**2 / w**4
    )

class make_cantilever_beam(core.model):
    def __init__(self):
        super().__init__(
            name="Cantilever Beam",
            # function=lambda x: function_beam(x),
            # outputs=["c_area", "g_stress", "g_displacement"],
            functions=[
                core.function(
                    function_area,
                    ["w", "t"],
                    ["c_area"],
                    "cross-sectional area"
                ),
                core.function(
                    function_stress,
                    ["w", "t", "H", "V", "E", "Y"],
                    ["g_stress"],
                    "limit state: stress"
                ),
                core.function(
                    function_displacement,
                    ["w", "t", "H", "V", "E", "Y"],
                    ["g_displacement"],
                    "limit state: tip displacement"
                )
            ],
            domain=core.domain(bounds={"w": [2, 4], "t": [2, 4]}),
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
