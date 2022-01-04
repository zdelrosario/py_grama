__all__ = ["make_trajectory_linear"]

from grama import cp_bounds, cp_vec_function, Model, df_make
from numpy import exp, Inf


## Constants
x0 = 0  # Initial x-position (m)
y0 = 0  # Initial y-position (m)
g = -9.8  # Gravitational acceleration (m/s^2)

## Responses (x and y trajectory components)
def fun_x(df):
    return df_make(
        x=df.tau * df.u0 * (1 - exp(-df.t / df.tau)) + x0
    )


def fun_y(df):
    v_inf = g * df.tau
    return df_make(
        y=df.tau * (df.v0 - v_inf) * (1 - exp(-df.t / df.tau)) + v_inf * df.t + y0
    )


# Units     (m/s) (m/s) (s)    (s)
var_list = ["u0", "v0", "tau", "t"]


def make_trajectory_linear():
    ## Assemble model
    md_trajectory = (
        Model("Trajectory Model")
        >> cp_vec_function(fun=fun_x, var=var_list, out=["x"], name="x_trajectory",)
        >> cp_vec_function(fun=fun_y, var=var_list, out=["y"], name="y_trajectory",)
        >> cp_bounds(
            u0=[0.1, Inf], v0=[0.1, Inf], tau=[0.05, Inf], t=[0, 600]
        )
    )

    return md_trajectory
