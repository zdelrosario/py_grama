from .. import core
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
        D_MAX - 4 * LENGTH**3 / E / w / t * sqrt(
            V**2 / t**4 + H**2 / w**4
        )
    ])

domain_cantilever_beam = core.domain_(
    hypercube = True,
    inputs    = ["w", "t", "H", "V", "E", "Y"],
    bounds    = {
        "w": [2., 4.],
        "t": [2., 4.],
        "H": [-Inf, +Inf],
        "V": [-Inf, +Inf],
        "E": [-Inf, +Inf],
        "Y": [-Inf, +Inf]
    }
)

density_cantilever_beam = core.density_(
    pdf = lambda X: 0.25 * \
            norm.pdf(X[2], loc = MU_H, scale = TAU_H) * \
            norm.pdf(X[3], loc = MU_V, scale = TAU_V) * \
            norm.pdf(X[4], loc = MU_E, scale = TAU_E) * \
            norm.pdf(X[5], loc = MU_Y, scale = TAU_Y),
    pdf_factors = ["unif", "unif", "norm", "norm", "norm", "norm"],
    pdf_param   = [
        {"lower": 2, "upper": 4},
        {"lower": 2, "upper": 4},
        {"loc": MU_H, "scale": TAU_H},
        {"loc": MU_V, "scale": TAU_V},
        {"loc": MU_E, "scale": TAU_E},
        {"loc": MU_Y, "scale": TAU_Y}
    ]
)

model_cantilever_beam = core.model_(
    function = function_beam,
    outputs  = ["c", "g_stress", "g_displacement"],
    domain   = domain_cantilever_beam,
    density  = density_cantilever_beam
)
