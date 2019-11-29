## Simple model pipeline example
import numpy as np
import pandas as pd
import grama as gr

from grama.models import model_cantilever_beam

# Setup
np.random.seed(101) # Set for reproducibility
n_monte_carlo = int(1e4)

# Instantiate model with desired geometry
model_base = model_cantilever_beam(w=2.80, t=3.0)

# Generate metamodel
model_meta = model_base >> gr.cp_metamodel(n=100)

# Draw samples
df_res = model_base >> gr.ev_monte_carlo(n_samples=n_monte_carlo)
df_res = df_res.rename(
    columns={"g_stress": "true_g_stress",
             "g_displacement": "true_g_displacement"}
)
df_res = model_meta >> gr.ev_df(df=df_res)

df_test = model_meta >> gr.ev_monte_carlo(n_samples=1)

# Compute output summary
print(df_res[
    ["true_g_stress", "g_stress", "true_g_displacement", "g_displacement"]
].describe())
