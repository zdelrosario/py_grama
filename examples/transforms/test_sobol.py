import numpy as np
import pandas as pd
import grama as gr
import re
from dfply import *

from grama.models import model_ishigami

## Ground truth for ishigami
a = 7.0; b = 0.1

D = (a**2)/8 + b*(np.pi**4)/5 + (b**2)*(np.pi**8)/18 + 0.5

S1 = (0.5 * (1 + b * np.pi**4 / 5)**2) / D
S2 = (a**2 / 8) / D
S3 = 0

T1 = (0.5 * (1 + b * np.pi**4 / 5)**2 + 8 * b**2 * np.pi**8 / 225) / D
T2 = (a**2 / 8) / D
T3 = (8 * b**2 * np.pi**8 / 225) / D

## Notebook parameters
n = int(1e3)

## Model setup
model = model_ishigami(a=a, b=b)

## First-order indices
df_first = \
    model >> \
    gr.ev_hybrid(n_samples=n) >> \
    gr.tf_sobol()

I_first = list(map(lambda s: s[0] == "S", df_first["var"]))
df_first = df_first[I_first]

print(df_first)
print([S1, S2, S3])

## Total-order indices
df_total = \
    model >> \
    gr.ev_hybrid(n_samples=n, plan="total") >> \
    gr.tf_sobol(plan="total")

I_total = list(map(lambda s: s[0] == "T", df_total["var"]))
df_total = df_total[I_total]

print(df_total)
print([T1, T2, T3])
