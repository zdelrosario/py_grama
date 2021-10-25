__all__ = ["make_poly"]

from grama import cp_copula_independence, cp_function, cp_marginals, Model


def make_poly():
    md = Model("Polynomials") >> \
         cp_function(fun=lambda x: x, var=1, out=1, name="linear") >> \
         cp_function(fun=lambda x: x**2, var=1, out=1, name="quadratic") >> \
         cp_function(fun=lambda x: x**3, var=1, out=1, name="cubic") >> \
         cp_marginals(
             x0={"dist": "uniform", "loc": -1, "scale": 2},
             x1={"dist": "uniform", "loc": -1, "scale": 2},
             x2={"dist": "uniform", "loc": -1, "scale": 2}
         ) >> \
         cp_copula_independence()

    return md
