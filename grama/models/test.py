__all__ = ["make_test", "df_test_input"]

import numpy as np
import pandas as pd
from .. import core

## Define a test dataset
df_test_input = pd.DataFrame(data={"x1": [0], "x2": [0], "x3": [0]})

## Define a test model
def fcn(x):
    x1, x2, x3 = x
    return x1 + x2 + x3

class make_test(core.Model):
    def __init__(self):
        super().__init__(
            name="test",
            functions=[
                core.Function(
                    fcn,
                    ["x1", "x2", "x3"],
                    ["f"],
                    "test"
                )
            ],
            domain=core.Domain(
                bounds={
                    "x1": [-1, +1],
                    "x2": [-1, +1],
                    "x3": [-1, +1]
                }
            ),
            density=core.Density(
                marginals=[
                    core.MarginalNamed(
                        "x1",
                        d_name="uniform",
                        d_param={"loc": -1, "scale": +2}
                    ),
                    core.MarginalNamed(
                        "x2",
                        d_name="uniform",
                        d_param={"loc": -1, "scale": +2}
                    )
                ]
            )
        )
