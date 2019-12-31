__all__ = ["make_ishigami"]

import numpy as np

import grama as gr

def fun(x):
    a, b, x1, x2, x3 = x
    return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

def make_ishigami():
    md = gr.Model(name = "Ishigami Function") >> \
        gr.cp_function(
            fun=fun,
            var=["a", "b", "x1", "x2", "x3"],
            out=1
        ) >> \
        gr.cp_bounds(a=(6.0, 8.0), b=(0, 0.2)) >> \
        gr.cp_marginals(
            x1={"dist": "uniform", "loc": -np.pi, "scale": 2 * np.pi},
            x2={"dist": "uniform", "loc": -np.pi, "scale": 2 * np.pi},
            x3={"dist": "uniform", "loc": -np.pi, "scale": 2 * np.pi}
        )

    return md
