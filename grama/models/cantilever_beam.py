__all__ = ["make_cantilever_beam"]

import numpy as np
import grama as gr
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

def function_area(x):
    w, t = x
    return w * t

def function_stress(x):
    w, t, H, V, E, Y = x
    return (Y - 600 * V / w / t**2 - 600 * H / w**2 / t) / MU_Y

def function_displacement(x):
    w, t, H, V, E, Y = x
    return D_MAX - np.float64(4) * LENGTH**3 / E / w / t * sqrt(
        V**2 / t**4 + H**2 / w**4
    )

def make_cantilever_beam():
    """Cantilever beam

    A standard reliability test-case, often used for benchmarking reliability
    analysis and design algorithms.

    Generally used in the following optimization problem:

        min_{w,t} c_area

        s.t.      P[g_stress <= 0] <= 1.35e-3

                  P[g_disp <= 0] <= 1.35e-3

                  1 <= w, t <= 4

    Deterministic Variables:
        w: Beam width
        t: Beam thickness
    Random Variables:
        H: Horizontal applied force
        V: Vertical applied force
        E: Elastic modulus
        Y: Yield stress
    Outputs:
        c_area: Cost; beam cross-sectional area
        g_stress: Limit state; stress
        g_disp: Limit state; tip displacement

    References:
        Wu, Y.-T., Shin, Y., Sues, R., and Cesare, M., "Safety-factor based approach for probability-based design optimization," American Institute of Aeronautics and Astronautics, Seattle, Washington, April 2001.
        Sues, R., Aminpour, M., and Shin, Y., "Reliability-based Multi-Disciplinary Optimiation for Aerospace Systems," American Institute of Aeronautics and Astronautics, Seattle, Washington, April 2001.

    """

    md = gr.Model(name = "Cantilever Beam") >> \
         gr.cp_function(
             fun=function_area,
             var=["w", "t"],
             out=["c_area"],
             name="cross-sectional area",
             runtime=1.717e-7
         ) >> \
         gr.cp_function(
             fun=function_stress,
             var=["w", "t", "H", "V", "E", "Y"],
             out=["g_stress"],
             name="limit state: stress",
             runtime=8.88e-7
         ) >> \
         gr.cp_function(
             fun=function_displacement,
             var=["w", "t", "H", "V", "E", "Y"],
             out=["g_disp"],
             name="limit state: displacement",
             runtime=3.97e-6
         ) >> \
         gr.cp_bounds(
             w=(2, 4),
             t=(2, 4)
         ) >> \
         gr.cp_marginals(
             H={"dist": "norm", "loc": MU_H, "scale": TAU_H, "sign": +1},
             V={"dist": "norm", "loc": MU_V, "scale": TAU_V, "sign": +1},
             E={"dist": "norm", "loc": MU_E, "scale": TAU_E, "sign":  0},
             Y={"dist": "norm", "loc": MU_Y, "scale": TAU_Y, "sign": -1}
         ) >> \
         gr.cp_copula_independence()

    return md
