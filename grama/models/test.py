__all__ = ["make_test", "df_test_input"]

from grama import (
    cp_bounds,
    cp_copula_independence,
    cp_function,
    cp_marginals,
    Model,
)
from pandas import DataFrame


## Define a test dataset
df_test_input = DataFrame(data={"x0": [0], "x1": [0], "x2": [0]})


## Define a test model
def fun(x1, x2, x3):
    return x1 + x2 + x3


def make_test():
    md = (
        Model()
        >> cp_function(fun=fun, var=3, out=1)
        >> cp_bounds(x0=(-1, +1), x1=(-1, +1), x2=(-1, +1))
        >> cp_marginals(
            x0={"dist": "uniform", "loc": -1, "scale": 2},
            x1={"dist": "uniform", "loc": -1, "scale": 2},
        )
        >> cp_copula_independence()
    )

    return md
