## Compare sampling plans
import numpy as np
import pandas as pd
import grama as gr
import time

from grama.models import model_composite_plate_tension

n = int(1e3)

# model = model_composite_plate_tension([0])
model = model_composite_plate_tension([-np.pi/4, +np.pi/4])

df_res = model >> gr.ev_lhs(n_samples=n, append=False)

print(df_res.describe())

pr_reliable = df_res.apply(lambda g: g > 0).mean()
print(pr_reliable)
