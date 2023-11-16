__all__ = ["make_piston", "make_piston_rand"]

from grama import cp_bounds, cp_copula_independence, cp_vec_function, cp_marginals, \
    Model, df_make, marg_mom
from numpy import sqrt, array, Inf, float64, pi


def make_piston():
    """Piston cycle time

    A simple model for the cycle time for a reciprocating piston within a cylinder. All inputs are treated as deterministic variables.

    Deterministic Variables:
        m: Piston weight (kg)
        s: Piston surface area (m^2)
        v_0: Initial gas volume (m^3)
        k: Spring coefficient (N/m)
        p_0: Atmospheric pressure (N/m^2)
        t_a: Ambient temperature (K)
        t_0: Filling gas temperature (K)

    Random Variables:
        (None)

    Outputs:
        t_cyc: cycle time (s)

    References:
        Kenett, R., & Zacks, S. (1998). Modern industrial statistics: design and control of quality and reliability. Pacific Grove, CA: Duxbury press.
        Moon, H. (2010). Design and Analysis of Computer Experiments for Screening Input Variables (Doctoral dissertation, Ohio State University).

    """
    md = (
        Model(name = "Piston cycle time")
        >> cp_vec_function(
            fun=lambda df: df_make(
                a=df.p_0 * df.s
                 +19.62 * df.m
                 -df.k * df.v_0 / df.s
            ),
            var=["p_0", "s", "m", "k", "v_0"],
            out=["a"],
            name="Intermediate calculation 1",
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                v=0.5*df.s/df.k * (
                    sqrt(df.a**2 + 4*df.k*df.p_0*df.v_0/df.t_0*df.t_a) - df.a
                )
            ),
            var=["s", "k", "a", "k", "p_0", "v_0", "t_0", "t_a"],
            out=["v"],
            name="Intermediate calculation 2",
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                t_cyc=2*pi*sqrt(
                    df.m / (df.k + df.s**2 * df.p_0*df.v_0/df.t_0 * df.t_a/df.v**2)
                )
            ),
            var=["m", "k", "s", "p_0", "v_0", "t_0", "t_a", "v"],
            out=["t_cyc"],
            name="Cycle time",
        )
        >> cp_bounds(
            m=(30, 60),          # kg
            s=(0.005, 0.020),    # m^2
            v_0=(0.002, 0.010),  # m^3
            k=(1000, 5000),      # N/m
            p_0=(90000, 110000), # Pa
            t_a=(290, 296),      # K
            t_0=(340, 360),      # K
        )
    )

    return md

def make_piston_rand():
    """Piston cycle time

    A simple model for the cycle time for a reciprocating piston within a cylinder. Some inputs are treated as deterministic nominal conditions with additive randomness.

    Deterministic Variables:
        m: Piston weight (kg)
        s: Piston surface area (m^2)
        v_0: Initial gas volume (m^3)

        k:   (Nominal) Spring coefficient (N/m)
        p_0: (Nominal) Atmospheric pressure (N/m^2)
        t_a: (Nominal) Ambient temperature (K)
        t_0: (Nominal) Filling gas temperature (K)

    Random Variables:
        dk:   Fluctuation in k   (unitless)
        dp_0: Fluctuation in p_0 (unitless)
        dt_a: Fluctuation in t_a (unitless)
        dt_0: Fluctuation in t_0 (unitless)

    Outputs:
        t_cyc: cycle time (s)

    References:
        Kenett, R., & Zacks, S. (1998). Modern industrial statistics: design and control of quality and reliability. Pacific Grove, CA: Duxbury press.
        Moon, H. (2010). Design and Analysis of Computer Experiments for Screening Input Variables (Doctoral dissertation, Ohio State University).

    """
    md = (
        Model(name = "Piston cycle time, with variability")
        >> cp_vec_function(
            fun=lambda df: df_make(
                rk=df.k * (1 + df.dk),
                rp_0=df.p_0 * (1 + df.dp_0),
                rt_a=df.t_a * (1 + df.dt_a),
                rt_0=df.t_0 * (1 + df.dt_0),
            ),
            var=["k", "p_0", "t_a", "t_0",
                 "dk", "dp_0", "dt_a", "dt_0"],
            out=["rk", "rp_0", "rt_a", "rt_0"],
            name="Random operating disturbances",
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                a=df.rp_0 * df.s
                 +19.62 * df.m
                 -df.rk * df.v_0 / df.s
            ),
            var=["rp_0", "s", "m", "rk", "v_0"],
            out=["a"],
            name="Intermediate calculation 1",
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                v=0.5*df.s/df.rk * (
                    sqrt(df.a**2 + 4*df.rk*df.rp_0*df.v_0/df.rt_0*df.rt_a) - df.a
                )
            ),
            var=["s", "a", "rk", "rp_0", "v_0", "rt_0", "rt_a"],
            out=["v"],
            name="Intermediate calculation 2",
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                t_cyc=2*pi*sqrt(
                    df.m / (df.rk + df.s**2 * df.rp_0*df.v_0/df.rt_0 * df.rt_a/df.v**2)
                )
            ),
            var=["m", "rk", "s", "rp_0", "v_0", "rt_0", "rt_a", "v"],
            out=["t_cyc"],
            name="Cycle time",
        )
        >> cp_bounds(
            m=(30, 60),          # kg
            s=(0.005, 0.020),    # m^2
            v_0=(0.002, 0.010),  # m^3
            k=(1000, 5000),      # N/m
            p_0=(90000, 110000), # Pa
            t_a=(290, 296),      # K
            t_0=(340, 360),      # K
        )
        >> cp_marginals(
            dk=marg_mom("uniform", mean=0, sd=0.40),
            dp_0=marg_mom("uniform", mean=0, sd=0.40),
            dt_a=marg_mom("uniform", mean=0, sd=0.40),
            dt_0=marg_mom("uniform", mean=0, sd=0.40),
        )
        >> cp_copula_independence()
    )

    return md
