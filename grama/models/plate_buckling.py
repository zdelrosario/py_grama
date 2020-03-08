__all__ = ["make_plate_buckle"]

import numpy as np
import grama as gr
from grama.data import df_stang

LOAD = 0.00128  # Applied load (kips)

## Reference geometry
THICKNESS = 0.06  # Plate thickness (in)
HEIGHT = 12.0  # Plate height    (in)


def function_buckle_state(x):
    t, h, w, E, mu, L = x
    return np.pi * E / 12 / (1 - mu ** 2) * (t / h) ** 2 - L / t / w


def make_plate_buckle():
    md = (
        gr.Model("Plate Buckling")
        >> gr.cp_function(
            fun=function_buckle_state,
            var=["t", "h", "w", "E", "mu", "L"],
            out=["g_buckle"],
            name="limit state",
        )
        >> gr.cp_bounds(
            t=(0.5 * THICKNESS, 2 * THICKNESS),
            h=(6, 18),
            w=(6, 18),
            L=(LOAD / 2, LOAD * 2),
        )
        >> gr.cp_marginals(
            E=gr.marg_named(df_stang.E, "norm"), mu=gr.marg_named(df_stang.mu, "beta")
        )
        >> gr.cp_copula_gaussian(df_data=df_stang)
    )

    return md
