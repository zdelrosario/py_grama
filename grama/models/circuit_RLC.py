__all__ = ["make_prlc", "make_prlc_rand"]

from grama import cp_bounds, cp_copula_independence, cp_function, \
    cp_bounds, cp_marginals, cp_vec_function, df_make, Model
from numpy import sqrt


## Component tolerances
C_percent_lo = -0.20
C_percent_up = +0.80
R_percent_lo = -0.05
R_percent_up = +0.05
L_percent_lo = -0.10
L_percent_up = +0.10

def make_prlc():
    md_RLC_det = (
        Model("RLC Circuit")
        >> cp_vec_function(
            fun=lambda df: df_make(omega0=sqrt(1 / df.L / df.C)),
            var=["L", "C"],
            out=["omega0"],
        )
        >> cp_function(
            fun=lambda df: df_make(Q=df.omega0 * df.R * df.C),
            name="parallel RLC",
            var=["omega0", "R", "C"],
            out=["Q"]
        )
        >> cp_bounds(
            R=(1e-3, 1e0),
            L=(1e-9, 1e-3),
            C=(1e-3, 100),
        )
    )

    return md_RLC_det

def make_prlc_rand():
    md_RLC_rand = (
        Model("RLC with component tolerances")
        >> cp_vec_function(
            fun=lambda df: df_make(
                Rr=df.R * (1 + df.dR),
                Lr=df.L * (1 + df.dL),
                Cr=df.C * (1 + df.dC),
            ),
            var=["R", "dR", "L", "dL", "C", "dC"],
            out=["Rr", "Lr", "Cr"],
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                omega0=sqrt(1 / df.Lr / df.Cr)
            ),
            var=["Lr", "Cr"],
            out=["omega0"],
        )
        >> cp_vec_function(
            fun=lambda df: df_make(Q=df.omega0 * df.Rr * df.Cr),
            name="parallel RLC",
            var=["omega0", "Rr", "Cr"],
            out=["Q"]
        )
        >> cp_bounds(
            R=(1e-3, 1e0),
            L=(1e-9, 1e-3),
            C=(1e-3, 100),
        )
        >> cp_marginals(
            dR=dict(
                dist="uniform",
                loc=R_percent_lo,
                scale=R_percent_up - R_percent_lo
            ),
            dL=dict(
                dist="uniform",
                loc=L_percent_lo,
                scale=L_percent_up - L_percent_lo
            ),
            dC=dict(
                dist="uniform",
                loc=C_percent_lo,
                scale=C_percent_up - C_percent_lo
            ),
        )
        >> cp_copula_independence()
    )

    return md_RLC_rand
