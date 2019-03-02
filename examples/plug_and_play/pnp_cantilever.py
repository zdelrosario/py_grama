## Simple model pipeline example
import grama.core as gr
from grama.evals import eval_monte_carlo
import pandas as pd

from grama.core import pi # Import pipe
from grama.models import model_cantilever_beam

df_res = \
    model_cantilever_beam(w = 2.5, t = 3) |pi| \
    eval_monte_carlo(n_samples = 10)

print(df_res)
