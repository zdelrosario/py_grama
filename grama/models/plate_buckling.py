__all__ = ["make_plate_buckle"]

from grama import cp_bounds, cp_copula_gaussian, cp_function, cp_marginals, \
    marg_named, Model
from grama.data import df_stang
from numpy import pi


LOAD = 0.00128  # Applied load (kips)

## Reference geometry
THICKNESS = 0.06  # Plate thickness (in)
HEIGHT = 12.0  # Plate height    (in)


def function_buckle_state(x):
    t, h, w, E, mu, L = x
    return pi * E / 12 / (1 - mu ** 2) * (t / h) ** 2 - L / t / w


def make_plate_buckle():
    md = (
        Model("Plate Buckling")
        >> cp_function(
            fun=function_buckle_state,
            var=["t", "h", "w", "E", "mu", "L"],
            out=["g_buckle"],
            name="limit state",
        )
        >> cp_bounds(
            t=(0.5 * THICKNESS, 2 * THICKNESS),
            h=(6, 18),
            w=(6, 18),
            L=(LOAD / 2, LOAD * 2),
        )
        >> cp_marginals(
            E=marg_named(df_stang.E, "norm"), mu=marg_named(df_stang.mu, "beta")
        )
        >> cp_copula_gaussian(df_data=df_stang)
    )

    return md
