__all__ = ["make_channel_nondim", "make_channel"]

from grama import cp_bounds, cp_copula_independence, cp_vec_function, cp_marginals
from grama import Model, df_make, marg_mom
from numpy import exp
from pandas import DataFrame

## Define the PSAAP 1d Channel model; dimensionless form
def make_channel_nondim():
    r"""Make 1d channel model; dimensionless form

    Instantiates a model for particle and fluid temperature rise; particles are suspended in a fluid with bulk velocity along a square cross-section channel. The walls of said channel are transparent, and radiation heats the particles as they travel down the channel.

    References:
        Banko, A.J. "RADIATION ABSORPTION BY INERTIAL PARTICLES IN A TURBULENT SQUARE DUCT FLOW" (2018) PhD Thesis, Stanford University, Chapter 2

    """
    md = (
        Model("1d Particle-laden Channel with Radiation; Dimensionless Form")
        >> cp_vec_function(
            fun=lambda df: df_make(
                beta=120 * (1 + df.Phi_M * df.chi)
            ),
            var=["Phi_M", "chi"],
            out=["beta"],
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                T_f=(df.Phi_M * df.chi) /
                    (1 + df.Phi_M * df.chi) * (
                        df.I * df.xst - df.beta**(-1) * df.I * (1 - exp(-df.beta * df.xst))
                    ),
                T_p=1 /
                    (1 + df.Phi_M * df.chi) * (
                        df.Phi_M * df.chi * df.I * df.xst
                      + df.beta**(-1) * df.I * (1 - exp(-df.beta * df.xst))
                    ),
            ),
            var=["xst", "Phi_M", "chi", "I", "beta"],
            out=["T_f", "T_p"],
        )
        >> cp_bounds(
            ## Dimensionless axial location (-)
            xst=(0, 5),
        )
        >> cp_marginals(
            ## Mass loading ratio (-)
            Phi_M={"dist": "uniform", "loc": 0, "scale": 1},
            ## Particle-fluid heat capacity ratio (-)
            chi={"dist": "uniform", "loc": 0.1, "scale": 0.9},
            ## Normalized radiative intensity (-)
            I={"dist": "uniform", "loc": 0.1, "scale": 0.9},
         )
        >> cp_copula_independence()
    )

    return md

## Define the PSAAP 1d Channel model; dimensional form
def make_channel():
    r"""Make 1d channel model; dimensional form

    Instantiates a model for particle and fluid temperature rise; particles are suspended in a fluid with bulk velocity along a square cross-section channel. The walls of said channel are transparent, and radiation heats the particles as they travel down the channel.

    Note that this takes the same inputs as the builtin dataset `df_channel`.

    References:
        Banko, A.J. "RADIATION ABSORPTION BY INERTIAL PARTICLES IN A TURBULENT SQUARE DUCT FLOW" (2018) PhD Thesis, Stanford University, Chapter 2

    Examples:

    >>> import grama as gr
    >>> from grama.data import df_channel
    >>> from grama.models import make_channel
    >>> md_channel = make_channel()

    >>> (
    >>>     df_channel
    >>>     >> gr.tf_md(md_channel)

    >>>     >> gr.ggplot(gr.aes("T_f", "T_norm"))
    >>>     + gr.geom_abline(slope=1, intercept=0, linetype="dashed")
    >>>     + gr.geom_point()
    >>>     + gr.labs(x="1D Model", y="3D DNS")
    >>> )

    """
    md = (
        Model("1d Particle-laden Channel with Radiation; Dimensional Form")
        >> cp_vec_function(
            fun=lambda df: df_make(
                Re=df.U * df.H / df.nu_f,
                chi=df.cp_p / df.cp_f,
                Pr=df.nu_f / df.alpha_f,
                Phi_M=df.rho_p * 0.524 * df.d_p**3 * df.n / df.rho_f,
                tau_flow=df.L / df.U,
                tau_pt=(df.rho_p * df.cp_p * 0.318 * df.d_p) / df.h_p,
                tau_rad=(df.rho_p * df.cp_p * 0.667 * df.d_p * df.T_0)
                       /(df.Q_abs * 0.78 * df.I_0),
            ),
            var=[
                "U",        # Fluid bulk velocity
                "H",        # Channel width
                "nu_f",     # Fluid kinematic viscosity
                "cp_p",     # Particle isobaric heat capacity
                "cp_f",     # Fluid isobaric heat capacity
                "alpha_f",  # Fluid thermal diffusivity
                "rho_p",    # Particle density
                "rho_f",    # Fluid density
                "d_p",      # Particle diameter
                "n",        # Particle number density
                "h_p",      # Particle-to-gas convection coefficient
                "T_0",      # Initial temperature
                "Q_abs",    # Particle radiation absorption coefficient
                "I_0",      # Incident radiation
            ],
            out=[
                "Re",       # Reynolds number
                "Pr",       # Prandtl number
                "chi",      # Particle-fluid heat capacity ratio
                "Phi_M",    # Mass Loading Ratio
                "tau_flow", # Fluid residence time
                "tau_pt",   # Particle thermal time constant
                "tau_rad",  # Particle temperature doubling time (approximate)
            ],
            name="Dimensionless Numbers",
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                ## Let xi = x / L
                xst=(df.xi * df.L) / df.H / df.Re / df.Pr,
                ## Assume an optically-thin scenario; I/I_0 = 1
                Is=df.Re * df.Pr * (df.H / df.L) * (df.tau_flow / df.tau_rad) * 1,
                beta=df.Re * df.Pr * (df.H / df.L) * (df.tau_flow / df.tau_pt)
                    *(1 + df.Phi_M * df.chi),
            ),
            var=["xi", "chi", "H", "L", "Phi_M", "tau_flow", "tau_rad", "tau_pt"],
            out=[
                "xst",   # Flow-normalized channel axial location
                "Is",    # Normalized heat flux
                "beta",  # Spatial development coefficient
            ],
            name="Intermediate Dimensionless Numbers",
        )
        >> cp_vec_function(
            fun=lambda df: df_make(
                T_f=(df.Phi_M * df.chi) /
                    (1 + df.Phi_M * df.chi) * (
                        df.Is * df.xst - df.Is / df.beta * (1 - exp(-df.beta * df.xst))
                    ),
                T_p=1 / (1 + df.Phi_M * df.chi) * (
                        df.Phi_M * df.chi * df.Is * df.xst
                      + df.Is / df.beta * (1 - exp(-df.beta * df.xst))
                    ),
            ),
            var=["xst", "Phi_M", "chi", "Is", "beta"],
            out=["T_f", "T_p"],
        )
        >> cp_bounds(
            ## Normalized axial location; xi = x/L (-)
            xi=(0, 1),
        )
        >> cp_marginals(
            ## Channel width (m)
            H={"dist": "uniform", "loc": 0.038, "scale": 0.004},
            ## Channel length (m)
            L={"dist": "uniform", "loc": 0.152, "scale": 0.016},
            ## Fluid bulk velocity (m/s)
            U={"dist": "uniform", "loc": 1, "scale": 2.5},
            ## Fluid kinematic viscosity (m^2/s)
            nu_f={"dist": "uniform", "loc": 1.4e-5, "scale": 0.1e-5},
            ## Particle isobaric heat capacity (J/(kg K))
            cp_p={"dist": "uniform", "loc": 100, "scale": 900},
            ## Fluid isobaric heat capacity (J/(kg K))
            cp_f={"dist": "uniform", "loc": 1000, "scale": 1000},
            ## Fluid thermal diffusivity (m^2/s)
            alpha_f={"dist": "uniform", "loc": 50e-6, "scale": 50e-6},
            ## Particle density (kg / m^3)
            rho_p={"dist": "uniform", "loc": 1e3, "scale": 9e3},
            ## Fluid density (kg / m^3)
            rho_f={"dist": "uniform", "loc": 0.5, "scale": 1.0},
            ## Particle diameter (m)
            d_p={"dist": "uniform", "loc": 1e-6, "scale": 9e-6},
            ## Particle number density (1 / m^3)
            n={"dist": "uniform", "loc": 9.5e9, "scale": 1.0e9},
            ## Particle-to-gas convection coefficient (W / (m^2 K))
            h_p={"dist": "uniform", "loc": 1e3, "scale": 9e3},
            ## Initial temperature (K)
            T_0={"dist": "uniform", "loc": 285, "scale": 30},
            ## Particle radiation absorption coefficient (-)
            Q_abs={"dist": "uniform", "loc": 0.25, "scale": 0.50},
            ## Incident radiation (W/m^2)
            I_0={"dist": "uniform", "loc": 9.5e6, "scale": 1.0e6},
         )
        >> cp_copula_independence()
    )

    return md
