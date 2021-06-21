__all__ = ["make_trajectory_linear"]

import grama as gr
from numpy import exp, Inf


## Constants
x0 = 0  # Initial x-position (m)
y0 = 0  # Initial y-position (m)
g = -9.8  # Gravitational acceleration (m/s^2)

## Responses (x and y trajectory components)
def fun_x(x):
    u0, v0, tau, t = x
    return tau * u0 * (1 - exp(-t / tau)) + x0


def fun_y(x):
    u0, v0, tau, t = x
    v_inf = g * tau
    return tau * (v0 - v_inf) * (1 - exp(-t / tau)) + v_inf * t + y0


# Units     (m/s) (m/s) (s)    (s)
var_list = ["u0", "v0", "tau", "t"]


def make_trajectory_linear():
    ## Assemble model
    md_trajectory = (
        gr.Model("Trajectory Model")
        >> gr.cp_function(fun=fun_x, var=var_list, out=["x"], name="x_trajectory",)
        >> gr.cp_function(fun=fun_y, var=var_list, out=["y"], name="y_trajectory",)
        >> gr.cp_bounds(
            u0=[0.1, Inf], v0=[0.1, Inf], tau=[0.05, Inf], t=[0, 600]
        )
    )

    return md_trajectory
