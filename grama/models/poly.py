__all__ = ["make_poly"]

import numpy as np

import grama as gr

def make_poly():
    md = gr.Model("Polynomials") >> \
         gr.cp_function(fun=lambda x: x, var=1, out=1, name="linear") >> \
         gr.cp_function(fun=lambda x: x**2, var=1, out=1, name="quadratic") >> \
         gr.cp_function(fun=lambda x: x**3, var=1, out=1, name="cubic") >> \
         gr.cp_marginals(
             x0={"dist": "uniform", "loc": -1, "scale": 2},
             x1={"dist": "uniform", "loc": -1, "scale": 2},
             x2={"dist": "uniform", "loc": -1, "scale": 2}
         ) >> \
         gr.cp_copula_independence()

    return md
