import grama as gr
import numpy as np
import pandas as pd

df = pd.DataFrame(
    data={
        "x": [0, 1, 2],
        "y": [3, 4, 5]
    }
)

df_weights1 = pd.DataFrame(
    data={
        "x": [1],
        "y": [1],
        "w": ['a']
    }
)

df_weights2 = pd.DataFrame(
    data={
        "x": [1, 1],
        "y": [1,-1],
        "w": ['a', 'b']
    }
)

df_res1 = df >> gr.tf_inner(df_weights1, append=True)
print(df_res1)

df_res2 = df >> gr.tf_inner(df_weights2, append=True)
print(df_res2)
