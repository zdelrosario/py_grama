import grama as gr
import numpy as np
import pandas as pd

from grama.models import *

# model = make_cantilever_beam()
# model = make_test()
# model = make_ishigami()
# model = make_linear_normal()
# model = make_composite_plate_tension(Theta_nom=[+np.pi/4,-np.pi/4])
# model = make_plate_buckle()
model = make_poly()

## Ensure proper bookkeeping
model.printpretty()

## Test nominal settings
# df_det = model.det_nom()
# print(df_det)

## Nominal model evaluation
df_nom = gr.eval_nominal(model, df_det="nom")
print(df_nom)

## Nominal model evaluation
df_conservative = gr.eval_conservative(model, df_det="nom")
print(df_conservative)

## Test gradient eval
df_grad = gr.eval_grad_fd(model, df_base=df_nom)
print(df_grad)

## Test monte carlo
# df_mc = gr.eval_lhs(model, n=1e1, df_det="nom")
# print(df_mc)

## Test sinews
# df_sinews = gr.eval_sinews(model, df_det="nom")
# print(df_sinews)

## Test hybrid points
# df_hybrid = gr.eval_hybrid(model, df_det="nom")
# print(df_hybrid)
