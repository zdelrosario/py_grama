## Test gradient
import numpy as np
import grama as gr
import pandas as pd

from grama.models import model_poly

model = model_poly()

df_nom = pd.DataFrame(
    data = {
        "x0": [-1, 0, +1],
        "x1": [-1, 0, +1],
        "x2": [-1, 0, +1]
    }
)

df_grad = model >> \
    gr.ev_grad_fd(
        df_base=df_nom,
        append=True,
        h=np.array([1e-3, 1e-6, 1e-9])
    )

print(df_grad.iloc[:, 0:3])
print(df_grad.iloc[:, 3:6])
print(df_grad.iloc[:, 6:9])
