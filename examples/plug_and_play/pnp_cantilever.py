## Simple model pipeline example
import grama.core as gr
import pandas as pd

from grama.core import pi # Import pipe
from grama.models import model_cantilever_beam

df_test = pd.DataFrame(
    data = {
        "w": [3.],
        "t": [3.],
        "H": [500.],
        "V": [1000.],
        "E": [2.9e7],
        "Y": [4e5]
    }
)

df_res = \
    model_cantilever_beam |pi| \
    gr.eval_df(
        df = df_test
    )

print(df_res)
