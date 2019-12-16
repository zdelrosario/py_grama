__all__ = ["make_test", "df_test_input"]

import numpy as np
import pandas as pd
from .. import core
from .. import compositions as cp

## Define a test dataset
df_test_input = pd.DataFrame(data={"x0": [0], "x1": [0], "x2": [0]})

## Define a test model
def fun(x):
    x1, x2, x3 = x
    return x1 + x2 + x3

def make_test():
    md = core.Model() >> \
         cp.cp_function(fun=fun, var=3, out=1) >> \
         cp.cp_bounds(x0=(-1,+1), x1=(-1,+1), x2=(-1,+1)) >> \
         cp.cp_marginals(
             x0={"dist": "uniform", "loc": -1, "scale": 2},
             x1={"dist": "uniform", "loc": -1, "scale": 2}
         )

    return md
