## Compare sampling plans
import numpy as np
import pandas as pd
import grama.core as gr

from grama.core import pi # Import pipe
from grama.models import model_cantilever_beam
from grama.evals import ev_monte_carlo, ev_lhs

np.random.seed(101) # Set for reproducibility
n_monte_carlo = int(1e3)

## Instantiate model with desired geometry
model = model_cantilever_beam(w = 2.80, t = 3.)

## Simple monte carlo
df_res_smc = \
    model |pi| \
    ev_monte_carlo(n_samples = n_monte_carlo)

## Latin Hypercube Sample
df_res_lhs = \
    model |pi| \
    ev_lhs(n_samples = n_monte_carlo)

## Compare
print(df_res_smc.describe())
print(df_res_lhs.describe())
