__all__ = ["make_sir"]

import grama as gr

from scipy.integrate import solve_ivp
from scipy.special import lambertw
from scipy import real
from toolz import curry

DF = gr.Intention()

## Model setup
def sir_rhs(t, y, beta, gamma):
    r"""Right-hand side (RHS) of SIR ODE model

    Implements the evolution equations (right-hand side of ODE system) for the SIR compartmental model for disease transmission.

    Args:
        t (float): Time (unused)
        y (array-like): State variables, [S, I, R]
        beta (float): Infectivity parameter
        gamma (float): Recovery parameter

    Returns:
        array-like: Time-derivatives of state variables, [dSdt, dIdt, dRdt]

    """
    ## Unpack data
    N = sum(y) # Total population
    S, I, R = y

    ## Compute derivatives
    dSdt = -beta * I * S / N
    dIdt = +beta * I * S / N - gamma * I
    dRdt = +gamma * I

    ## Package for solver
    return [dSdt, dIdt, dRdt]

def sir_vtime(T, S0, I0, R0, beta, gamma, rtol=1e-4):
    r"""Solve SIR IVP, vectorized over T

    Solves the initial value problem (IVP) associated with the SIR model, given parameter values and a span of time values. This routine uses an adaptive timestep to solve the IVP to a specified tolerance, then uses cubic interpolation to query the time points of interest.

    Args:
        T (array-like): Time points of interest
        S0 (float): Initial number of susceptible individuals (at t=0)
        I0 (float): Initial number of infected individuals (at t=0)
        R0 (float): Initial number of removed individuals (at t=0)
        beta (float): Infection rate parameter
        gamma (float): Removal rate parameter

    Returns:
        pandas DataFrame: Simulation timeseries results
    """

    ## Solve SIR model on adaptive, coarse time mesh
    T_span = [0, max(T)]
    y0 = [S0, I0, R0]

    res = solve_ivp(
        sir_rhs,
        T_span,
        y0,
        args=(beta, gamma),
        rtol=rtol,
        t_eval=T,
    )

    ## Interpolate to desired T points
    df_res = gr.df_make(
        t=T,
        S=res.y[0, :],
        I=res.y[1, :],
        R=res.y[2, :],
        S0=[S0],
        I0=[I0],
        R0=[R0],
        beta=[beta],
        gamma=[gamma],
    )

    return df_res

@curry
def fun_sir(df, rtol=1e-4):
    r"""Fully-vectorized SIR solver

    SIR IVP solver, vectorized over parameter values **and** time. The routine identifies groups of parameter values and runs a vectorized IVP solver over all associated time points, and gathers all results into a single output DataFrame. Intended for use in a grama model.

    Args:
        df (pd.DataFrame): All input values; must contain columns for all SIR parameters

    Preconditions:
        ["t", "S0", "I0", "R0", "beta", "gamma"] in df.columns

    Postconditions:
        Row-ordering of input data is reflected in the row-ordering of the output.

    Returns:
        pd.DataFrame: Solution results
    """
    ## Find all groups of non-t parameters
    df_grouped = (
        df
        >> gr.tf_mutate(
            _idx=DF.index,
            _code=gr.str_c(
                "S0", DF.S0,
                "I0", DF.I0,
                "R0", DF.R0,
                "beta", DF.beta,
                "gamma", DF.gamma,
            )
        )
    )

    ## Run time-vectorized SIR solver over each group
    df_results = gr.df_grid()
    codes = set(df_grouped._code)
    for code in codes:
        df_param = (
            df_grouped
            >> gr.tf_filter(DF._code == code)
        )

        df_results = (
            df_results
            >> gr.tf_bind_rows(
                sir_vtime(
                    df_param.t,
                    df_param.S0[0],
                    df_param.I0[0],
                    df_param.R0[0],
                    df_param.beta[0],
                    df_param.gamma[0],
                    rtol=rtol,
                )
                >> gr.tf_mutate(_idx=df_param._idx)
            )
        )

    ## Sort to match original ordering
    # NOTE: Without this, the output rows will be scrambled, relative
    # to the input rows, leading to very confusing output!
    return (
        df_results
        >> gr.tf_arrange(DF._idx)
        >> gr.tf_drop("_idx")
    )

## Model builder
def make_sir(rtol=1e-4):
    r"""Make an SIR model

    Instantiates a Susceptible, Infected, Removed (SIR) model for disease transmission.

    Args:
        rtol (float): Relative tolerance for IVP solver

    Returns:
        grama Model: SIR model

    References:
        "Compartmental models in epidemiology," Wikipedia, url: https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

    """

    md_sir = (
        gr.Model("SIR Model")
        >> gr.cp_vec_function(
            fun=lambda df: gr.df_make(
                # Assume no recovered people at virus onset
                R0=0,
                # Assume total population of N=100
                S0=df.N - df.I0,
            ),
            var=["I0", "N"],
            out=["S0", "R0"],
            name="Population setup",
        )
        >> gr.cp_vec_function(
            fun=fun_sir(rtol=rtol),
            var=["t", "S0", "I0", "R0", "beta", "gamma"],
            out=["S", "I", "R"],
            name="ODE solver & interpolation",
        )
        >> gr.cp_bounds(
            N=(100, 100), # Fixed population size
            I0=(1, 10),
            beta=(0.1, 0.5),
            gamma=(0.1, 0.5),
            t=(0, 100),
        )
    )

    return md_sir
