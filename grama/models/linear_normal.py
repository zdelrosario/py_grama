__all__ = ["make_linear_normal"]

from grama import cp_copula_independence, cp_function, cp_marginals, Model


def limit_state(x1, x2):
    return 1 - x1 - x2


def make_linear_normal():
    md = (
        Model("Linear-Normal Reliability Problem")
        >> cp_function(
            fun=limit_state, var=2, out=["g_linear"], name="limit state"
        )
        >> cp_marginals(
            x0={"dist": "norm", "loc": 0, "scale": 1, "sign": +1},
            x1={"dist": "norm", "loc": 0, "scale": 1, "sign": +1},
        )
        >> cp_copula_independence()
    )

    return md
