## Simple model pipeline example
import grama.core as gr
import pandas as pd

from grama.core import pi # Import pipe

model_default = gr.model_()
df_default    = pd.DataFrame(
    data = {"x" : [0, 1]}
)

df_res = \
    model_default |pi| \
    gr.eval_df(
        df = df_default
    )

print(df_res)
