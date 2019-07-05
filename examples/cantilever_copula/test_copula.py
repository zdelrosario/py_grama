## Simple model pipeline example
import numpy as np
import pandas as pd
import grama.core as gr

from grama.core import pi # Import pipe
from grama.models import model_cantilever_beam
from grama.evals import *

np.random.seed(101) # Set for reproducibility

n_monte_carlo = int(1e4)

## Instantiate model with desired geometry
model_indep  = model_cantilever_beam(w = 2.80, t = 3.)
model_copula = model_cantilever_beam(w = 2.80, t = 3.)

## Modify model to introduce copula structure
n_in   = len(model_copula.density.pdf_factors)
n_corr = len(np.triu_indices(n_in, 1)[0])

model_copula.density.pdf_corr = [0.1] * n_corr

# ## DEBUG nominal vs conservative
# df_nom = model_indep |pi|\
#     ev_nominal()

# df_con = model_indep |pi|\
#     ev_conservative()

# print(df_nom)
# print(df_con)

# Draw samples
df_res_indep  = model_indep |pi| \
    ev_monte_carlo(n_samples = n_monte_carlo)

df_res_copula = model_copula |pi| \
    ev_monte_carlo(n_samples = n_monte_carlo)

# Compare input marginals
print(df_res_indep[ ["H", "V", "E", "Y"]].describe())
print(df_res_copula[["H", "V", "E", "Y"]].describe())
## Note, marginal summary stats look similar

# Compute output summary
print(df_res_indep[ ["g_stress", "g_displacement"]].describe())
print(df_res_copula[["g_stress", "g_displacement"]].describe())
## Note, output summary stats look very dissimilar!
