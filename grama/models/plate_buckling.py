__all__ = ["make_plate_buckle"]

from grama import cp_bounds, cp_copula_gaussian, cp_vec_function, cp_marginals, \
    marg_fit, Model, df_make
from grama.data import df_stang
from numpy import pi


LOAD = 0.00128  # Applied load (kips)

## Reference geometry
THICKNESS = 0.06  # Plate thickness (in)
HEIGHT = 12.0  # Plate height    (in)


def make_plate_buckle():
    r"""Initialize a buckling plate model

    Variables (deterministic):
        w (in): Plate width
        h (in): Plate height
        t (in): Plate thickness
        m (-): Wavenumber
        L (kips): Applied (compressive) load;
            uniformly applied along top and bottom edges

    Variables (random):
        E (kips/in^2): Elasticity
        mu (-): Poisson's ratio

    Outputs:
        k_cr (-): Prefactor for buckling stress
        g_buckle (kips/in^2): Buckling limit state:
            critical stress - applied stress
    """
    md = (
        Model("Plate Buckling")
        >> cp_vec_function(
            fun=lambda df: df_make(
                k_cr=(df.m*df.h/df.w + df.w/df.m/df.h)**2
            ),
            var=["w", "h", "m"],
            out=["k_cr"],
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                g_buckle=df.k_cr * pi**2/12 * df.E / (1 - df.mu**2) * (df.t/df.h)**2
                - df.L / df.t / df.w
            ),
            var=["k_cr", "t", "h", "w", "E", "mu", "L"],
            out=["g_buckle"],
            name="limit state",
        )
        >> cp_bounds(
            t=(0.5 * THICKNESS, 2 * THICKNESS),
            h=(6, 18),
            w=(6, 18),
            m=(1, 5),
            L=(LOAD / 2, LOAD * 2),
        )
        >> cp_marginals(
            E=marg_fit("norm", df_stang.E),
            mu=marg_fit("beta", df_stang.mu),
        )
        >> cp_copula_gaussian(df_data=df_stang)
    )

    return md
