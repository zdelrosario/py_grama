## Simple model pipeline example
import numpy as np
import pandas as pd
import grama as gr

from grama.models import model_cantilever_beam

np.random.seed(101) # Set for reproducibility

n_monte_carlo = int(1e4)

## Instantiate model with desired geometry
model  = model_cantilever_beam(w=2.80, t=3.00)

df_res = gr.eval_sinews(model)

print(df_res[model.domain.inputs].describe())
print(df_res[model.outputs].describe())
