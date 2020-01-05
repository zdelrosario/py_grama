__all__ = ["make_linear_normal"]

import numpy as np
import grama as gr

def limit_state(x):
    x1, x2 = x

    return 1 - x1 - x2

def make_linear_normal():
    md = gr.Model("Linear-Normal Reliability Problem") >> \
         gr.cp_function(
             fun=limit_state,
             var=2,
             out=["g_linear"],
             name="limit state"
         ) >> \
         gr.cp_marginals(
             x0={"dist": "norm", "loc": 0, "scale": 1, "sign":+1},
             x1={"dist": "norm", "loc": 0, "scale": 1, "sign":+1}
         ) >> \
         gr.cp_copula_independence()

    return md
