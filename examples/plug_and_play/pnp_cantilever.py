## Simple model pipeline example
import numpy as np
import pandas as pd
import grama as gr

from grama import pi
from grama.models import model_cantilever_beam

np.random.seed(101) # Set for reproducibility

n_monte_carlo = int(1e3)
n_train       = int(50)

## Instantiate model with desired geometry
model = model_cantilever_beam(w = 2.80, t = 3.)

## Draw a number of MC samples
df_res_direct = \
    model |pi| \
    gr.ev_monte_carlo(n_samples = n_monte_carlo)

## Draw samples for training
df_res_train = \
    model |pi| \
    gr.ev_monte_carlo(n_samples = n_train)

## Fit a meta-model via OLS
## Mysteriously, the pipe does not work for ft_ols()...
model_fitted = gr.ft_ols(
    df_res_train,
    formulae = [
        "g_stress ~ H + V + E + Y",
        "g_displacement ~ H + V + E + Y"
    ],
    domain  = model.domain,
    density = model.density
)

## Draw more samples via the model
df_res_surrogate = \
    model_fitted |pi| \
    gr.ev_monte_carlo(n_samples = n_monte_carlo)

## Post-process
R_stress_direct       = np.mean(df_res_direct.loc[:, "g_stress"] >= 0)
R_displacement_direct = np.mean(df_res_direct.loc[:, "g_displacement"] >= 0)

R_stress_surrogate       = np.mean(df_res_surrogate.loc[:, "g_stress"] >= 0)
R_displacement_surrogate = np.mean(df_res_surrogate.loc[:, "g_displacement"] >= 0)

## Print training samples
print(df_res_train)

print("R_stress_direct    = {0:4.3e}".format(R_stress_direct))
print("R_stress_surrogate = {0:4.3e}".format(R_stress_surrogate))

print("R_displacement_direct    = {0:4.3e}".format(R_displacement_direct))
print("R_displacement_surrogate = {0:4.3e}".format(R_displacement_surrogate))
