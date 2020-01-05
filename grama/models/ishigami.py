__all__ = ["make_ishigami"]

import numpy as np

import grama as gr

def fun(x):
    a, b, x1, x2, x3 = x
    return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

def make_ishigami():
    """Ishigami function

    The Ishigami function is commonly used as a test case for estimating Sobol'
    indices.

    Model definition:

        y0 = sin(x1) + a sin(x2)^2 + b x3^4 sin(x1)

        x1 ~ U[-pi, +pi]

        x2 ~ U[-pi, +pi]

        x3 ~ U[-pi, +pi]

    Sobol' index data:

        V[y0] = a^2/8 + b pi^4/5 + b^2 pi^8/18 + 0.5

        T1 = 0.5(1 + b pi^4/5)^2

        T2 = a^2/8

        T3 = 0

        Tt1 = 0.5(1 + b pi^4/5)^2 + 8 b^2 pi^8/225

        Tt2 = a^2/8

        Tt3 = 8 b^2 pi^8/225

    References:
        T. Ishigami and T. Homma, “An importance quantification technique in uncertainty analysis for computer models,” In the First International Symposium on Uncertainty Modeling and Analysis, Maryland, USA, Dec. 3–5, 1990. DOI:10.1109/SUMA.1990.151285
    """

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
        ) >> \
        gr.cp_copula_independence()

    return md
