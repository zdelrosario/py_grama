## Simple model pipeline example
import numpy as np
import pandas as pd
import grama as gr

from grama import pi # Import pipe
from grama.models import model_cantilever_beam

np.random.seed(101) # Set for reproducibility

n_monte_carlo = int(1e4)

## Instantiate model with desired geometry
model  = model_cantilever_beam(w = 2.80, t = 3.)

df_res = gr.ev_sweeps_marginal(model)

# Compare input marginals
print(df_res[ ["H", "V", "E", "Y"]].describe())
## Note, marginal summary stats look similar

# Compute output summary
print(df_res[ ["g_stress", "g_displacement"]].describe())
## Note, output summary stats look very dissimilar!
