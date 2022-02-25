all = ["make_pareto_random"]

from grama import Model, cp_vec_function, cp_marginals, cp_copula_independence, \
    df_make, cos, sin, tan
from numpy import pi

def make_pareto_random(twoDim = True):
    """ Create a model of random points for a pareto frontier evaluation
    Args:
        twoDim (bool): determines whether to create a 2D or 3D model
    """
    if twoDim == True:
        # Model to make dataset
        md_true = (
            Model()
            >> cp_vec_function(
                fun=lambda df: df_make(
                    y1=df.x1 * cos(df.x2),
                    y2=df.x1 * sin(df.x2),
                ),
                var=["x1", "x2"],
                out=["y1", "y2"],
            )
            >> cp_marginals(
                x1=dict(dist="uniform", loc=0, scale=1),
                x2=dict(dist="uniform", loc=0, scale=pi/2),
            )
            >> cp_copula_independence()
        )

        return md_true
    else:
        # Model to make dataset
        md_true = (
            Model()
            >> cp_vec_function(
                fun=lambda df: df_make(
                    y1=df.x1 * cos(df.x2),
                    y2=df.x1 * sin(df.x2),
                    y3=df.x1 * tan(df.x2),
                ),
                var=["x1", "x2","x3"],
                out=["y1", "y2","y3"],
            )
            >> cp_marginals(
                x1=dict(dist="uniform", loc=0, scale=1),
                x2=dict(dist="uniform", loc=0, scale=pi/2),
                x3=dict(dist="uniform", loc=0, scale=pi/4)
            )
            >> cp_copula_independence()
        )

        return md_true
