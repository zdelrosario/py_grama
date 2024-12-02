__all__ = ["make_cantilever_beam"]

from grama import (
    cp_bounds,
    cp_copula_independence,
    cp_vec_function,
    cp_marginals,
    Model,
    df_make,
)
from collections import OrderedDict as od
from numpy import sqrt, array, Inf, float64
from scipy.stats import norm


LENGTH = 100
D_MAX = 2.2535

MU_H = 500.0
MU_V = 1000.0
MU_E = 2.9e7
MU_Y = 40000.0

TAU_H = 100.0
TAU_V = 100.0
TAU_E = 1.45e6
TAU_Y = 2000.0


def function_area(df):
    return df_make(c_area=df.w * df.t)


def function_stress(df):
    return df_make(
        g_stress=(
            df.Y - 600 * df.V / df.w / df.t**2 - 600 * df.H / df.w**2 / df.t
        )
        / MU_Y
    )


def function_displacement(df):
    return df_make(
        g_disp=D_MAX
        - float64(4)
        * LENGTH**3
        / df.E
        / df.w
        / df.t
        * sqrt(df.V**2 / df.t**4 + df.H**2 / df.w**4)
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

    md = (
        Model(name="Cantilever Beam")
        >> cp_vec_function(
            fun=function_area,
            var=["w", "t"],
            out=["c_area"],
            name="cross-sectional area",
            runtime=1.717e-7,
        )
        >> cp_vec_function(
            fun=function_stress,
            var=["w", "t", "H", "V", "E", "Y"],
            out=["g_stress"],
            name="limit state: stress",
            runtime=8.88e-7,
        )
        >> cp_vec_function(
            fun=function_displacement,
            var=["w", "t", "H", "V", "E", "Y"],
            out=["g_disp"],
            name="limit state: displacement",
            runtime=3.97e-6,
        )
        >> cp_bounds(w=(2, 4), t=(2, 4))
        >> cp_marginals(
            H={"dist": "norm", "loc": MU_H, "scale": TAU_H, "sign": +1},
            V={"dist": "norm", "loc": MU_V, "scale": TAU_V, "sign": +1},
            E={"dist": "norm", "loc": MU_E, "scale": TAU_E, "sign": 0},
            Y={"dist": "norm", "loc": MU_Y, "scale": TAU_Y, "sign": -1},
        )
        >> cp_copula_independence()
    )

    return md
