## Simple model pipeline example
import numpy as np
import pandas as pd
import grama as gr
import matplotlib.pyplot as plt

from grama.models import make_cantilever_beam

np.random.seed(101) # Set for reproducibility

n_monte_carlo = int(1e4)

## Instantiate model with desired geometry
model  = make_cantilever_beam()

df_res = gr.eval_sinews(model, df_det="nom")

print(df_res[model.var_rand].describe())
print(df_res[model.outputs].describe())

# df_res >> gr.pt_sinew_inputs(var=model.var_rand)
# plt.show()

# df_res >> gr.pt_sinew_outputs(var=model.var_rand, outputs=model.outputs)
# plt.show()

# df_res >> gr.pt_auto()
gr.plot_auto(df_res)
plt.show()
