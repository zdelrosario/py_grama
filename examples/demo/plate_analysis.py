import grama as gr
import numpy as np

## Load data for RV model
from grama.data import df_stang

## Functions
def fun_critical(x):
    E, mu, t, h = x
    return np.pi ** 2 * E / 12 / (1 - mu ** 2) * (t / h) ** 2


var_critical = ["E", "mu", "t", "h"]
out_critical = ["sig_cr"]


def fun_applied(x):
    L, w, t = x
    return L / w / t


var_applied = ["L", "w", "t"]
out_applied = ["sig_app"]


def fun_limit(x):
    sig_cr, sig_app = x
    return sig_cr - sig_app


var_limit = ["sig_cr", "sig_app"]
out_limit = ["safety"]

## Build model
md_plate = (
    gr.Model("Plate under buckling load")
    >> gr.cp_function(
        fun=fun_critical, var=var_critical, out=out_critical, name="Critical"
    )
    >> gr.cp_function(fun=fun_applied, var=var_applied, out=out_applied, name="Applied")
    >> gr.cp_function(fun=fun_limit, var=var_limit, out=out_limit, name="Safety")
    >> gr.cp_bounds(  # Deterministic variables
        t=(0.03, 0.12),  # Thickness
        w=(6, 18),  # Width
        h=(6, 18),  # Height
        L=(2.5e-1, 4.0e-1),  # Load
    )
    >> gr.cp_marginals(  # Random variables
        E=gr.marg_gkde(df_stang.E), mu=gr.marg_gkde(df_stang.mu)
    )
    >> gr.cp_copula_gaussian(df_data=df_stang)
)  # Dependence
