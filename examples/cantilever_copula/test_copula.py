## Simple model pipeline example
import numpy as np
import pandas as pd
import grama as gr

from grama.models import model_cantilever_beam

np.random.seed(101) # Set for reproducibility

n_monte_carlo = int(1e4)

## Instantiate model with desired geometry
model_indep  = model_cantilever_beam(w=2.80, t=3.)
model_copula = model_cantilever_beam(w=2.80, t=3.)

## Modify model to introduce copula structure
n_in   = len(model_copula.density.pdf_factors)
n_corr = len(np.triu_indices(n_in, 1)[0])

model_copula.density.pdf_corr = [0.1] * n_corr

# Draw samples
df_res_indep  = model_indep >> \
    gr.ev_monte_carlo(n_samples = n_monte_carlo)

df_res_copula = model_copula >> \
    gr.ev_monte_carlo(n_samples = n_monte_carlo)

# Compare input marginals
print(df_res_indep[ ["H", "V", "E", "Y"]].describe())
print(df_res_copula[["H", "V", "E", "Y"]].describe())
## Note, marginal summary stats look similar

# Compute output summary
print(df_res_indep[ ["g_stress", "g_displacement"]].describe())
print(df_res_copula[["g_stress", "g_displacement"]].describe())
## Note, output summary stats look very dissimilar!
