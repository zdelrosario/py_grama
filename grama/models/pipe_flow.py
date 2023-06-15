__all__ = ["make_pipe_friction"]

from grama import cp_bounds, cp_function, cp_marginals, Model, df_make
from numpy import array
from math import sqrt, log10, pow, log
from scipy.optimize import bisect

def re_fcn(rho,u,d,mu,eps):
    # {rho,u,d,mu,eps}
    return rho*u*d/mu

def f_lam(rho,u,d,mu,eps):
    # {rho,u,d,mu,eps}
    return 64. / re_fcn(rho,u,d,mu,eps)

def colebrook(rho,u,d,mu,eps,f):
    # {rho,u,d,mu,eps}
    fs = sqrt(f); Re = re_fcn(rho,u,d,mu,eps)
    return 1 + 2.*fs*log10(eps/3.6/d + 2.51/Re/fs)

def f_tur(rho,u,d,mu,eps):
    return bisect(lambda f: colebrook(rho,u,d,mu,eps,f), 1e-5, 10)

Re_c = 3e3
def fcn_pipe(rho,u,d,mu,eps):
    Re = re_fcn(rho,u,d,mu,eps)
    if Re < Re_c:
        return f_lam(rho,u,d,mu,eps)
    else:
        return f_tur(rho,u,d,mu,eps)

def make_pipe_friction():
    r"""Pipe Friction Factor

    Approximate the pipe friction factor via simple equations. In the laminar regime ($Re < 3,000$ assumed) the analytic Poiseuille equation is used. Beyond $Re = 3,000$ the empirical expressions of Colebrook and White are used.

    Deterministic variables:
        d: Pipe diameter (m)
        eps: Pipe roughness lengthscale (m)

        rho: Fluid density (kg / m^3)
        u: Fluid bulk velocity (m / s)
        mu: Fluid dynamic viscosity (Pa * s)

    Random Variables: None

    Outputs:
        f: friction factor (-)

    References:
        White, *Fluid Mechanics*, McGraw-Hill, New York, 2011
        Nikuradse, *Laws of Flow in Rough Pipes*, National Advisory Committee for Aeronautics, Washington, D.C., 1950
        del Rosario, Lee, and Iaccarino, "Lurking Variable Detection via Dimensional Analysis," *SIAM/ASA J. Uncertainty Quantification*, 2019

    """
    md = (
        Model("Pipe Friction Factor")
        >> cp_function(
            fun=fcn_pipe,
            var=["rho", "u", "d", "mu", "eps"],
            out=["f"]
        )
        >> cp_bounds(
            rho=(1.0, 1.4),
            u=(1e-4, 1e+1),
            d=(1.3, 1.7),
            mu=(1.0e-5, 1.5e-5),
            eps=(0.5e-1, 2.0e-1),
        )
    )

    return md
