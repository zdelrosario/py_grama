__all__ = [
    "eval_contour",
    "ev_contour",
]

from grama import eval_df, add_pipe, tf_outer
from numpy import array, linspace, isfinite, reshape, full
from pandas import concat, DataFrame
from toolz import curry

class Square():
    A = [0, 0]
    B = [0, 0]
    C = [0, 0]
    D = [0, 0]
    A_data = 0.0
    B_data = 0.0
    C_data = 0.0
    D_data = 0.0

    def GetCaseId(self, threshold):
        caseId = 0
        if (self.A_data >= threshold):
            caseId |= 1
        if (self.B_data >= threshold):
            caseId |= 2
        if (self.C_data >= threshold):
            caseId |= 4
        if (self.D_data >= threshold):
            caseId |= 8

        return caseId

    def GetLines(self, Threshold):
        lines = []
        caseId = self.GetCaseId(Threshold)

        if caseId in (0, 15):
            return []

        if caseId in (1, 14, 10):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.B[1]
            qX = self.D[0]
            qY = (self.A[1] + self.D[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (2, 13, 5):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.A[1]
            qX = self.C[0]
            qY = (self.A[1] + self.D[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (3, 12):
            pX = self.A[0]
            pY = (self.A[1] + self.D[1]) / 2
            qX = self.C[0]
            qY = (self.B[1] + self.C[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (4, 11, 10):
            pX = (self.C[0] + self.D[0]) / 2
            pY = self.D[1]
            qX = self.B[0]
            qY = (self.B[1] + self.C[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        elif caseId in (6, 9):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.A[1]
            qX = (self.C[0] + self.D[0]) / 2
            qY = self.C[1]

            line = (pX, pY, qX, qY)

            lines.append(line)

        elif caseId in (7, 8, 5):
            pX = (self.C[0] + self.D[0]) / 2
            pY = self.C[1]
            qX = self.A[0]
            qY = (self.A[1] + self.D[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        return lines

def marching_square(xVector, yVector, Data, threshold):
    linesList = []

    Height = len(Data)  # rows
    Width = len(Data[1])  # cols

    if ((Width == len(xVector)) and (Height == len(yVector))):
        squares = full((Height - 1, Width - 1), Square())

        sqHeight = squares.shape[0]  # rows count
        sqWidth = squares.shape[1]  # cols count

        for j in range(sqHeight):  # rows
            for i in range(sqWidth):  # cols
                a = Data[j + 1, i]
                b = Data[j + 1, i + 1]
                c = Data[j, i + 1]
                d = Data[j, i]
                A = [xVector[i], yVector[j + 1]]
                B = [xVector[i + 1], yVector[j + 1]]
                C = [xVector[i + 1], yVector[j]]
                D = [xVector[i], yVector[j]]

                squares[j, i].A_data = a
                squares[j, i].B_data = b
                squares[j, i].C_data = c
                squares[j, i].D_data = d

                squares[j, i].A = A
                squares[j, i].B = B
                squares[j, i].C = C
                squares[j, i].D = D

                listTemp = squares[j, i].GetLines(threshold)

                linesList = linesList + listTemp
    else:
        raise AssertionError

    return [linesList]

## Generate contours from a model
# --------------------------------------------------
@curry
def eval_contour(
        model,
        var=None,
        out=None,
        df=None,
        levels=None,
        n_side=128,
        n_levels=5,
):
    r"""Generate contours from a model

    Generates contours from a model. Evaluates the model on a dense grid, then
    runs marching squares to generate contours. Supports targeting multiple
    outputs and handling auxiliary inputs not included in the contour map.

    Args:
        model (gr.Model): Model to evaluate.
        var (list of str): Model inputs to target; must provide exactly
            two inputs, and both must have finite domain width.
        out (list of str): Model output(s) for contour generation.
        df (DataFrame): Levels for model variables not included in var (auxiliary inputs).
        levels (dict): Specific output levels for contour generation;
            overrides n_levels.
        n_side (int): Side resolution for grid; n_side**2 total evaluations.
        n_levels (int): Number of contour levels.

    Returns:
        DataFrame: Points along contours, organized by output and auxiliary variable levels.

    Examples:

        >>> import grama as gr
        >>> ## Multiple outputs
        >>> (
        >>>     gr.Model()
        >>>     >> gr.cp_vec_function(
        >>>         fun=lambda df: gr.df_make(
        >>>             f=df.x**2 + df.y**2,
        >>>             g=df.x + df.y,
        >>>         ),
        >>>         var=["x", "y"],
        >>>         out=["f", "g"],
        >>>     )
        >>>     >> gr.cp_bounds(
        >>>         x=(-1, +1),
        >>>         y=(-1, +1),
        >>>     )
        >>>     >> gr.ev_contour(
        >>>         var=["x", "y"],
        >>>         out=["f", "g"],
        >>>     )
        >>>
        >>>     >> gr.ggplot(gr.aes("x", "y"))
        >>>     + gr.geom_segment(gr.aes(xend="x_end", yend="y_end", group="level", color="out"))
        >>> )
        >>> ## Auxiliary inputs
        >>> (
        >>>     gr.Model()
        >>>     >> gr.cp_vec_function(
        >>>         fun=lambda df: gr.df_make(
        >>>             f=df.c * df.x + (1 - df.c) * df.y,
        >>>         ),
        >>>         var=["x", "y"],
        >>>         out=["f", "g"],
        >>>     )
        >>>     >> gr.cp_bounds(
        >>>         x=(-1, +1),
        >>>         y=(-1, +1),
        >>>     )
        >>>     >> gr.ev_contour(
        >>>         var=["x", "y"],
        >>>         out=["f"],
        >>>         df=gr.df_make(c=[0, 1])
        >>>     )
        >>>
        >>>     >> gr.ggplot(gr.aes("x", "y"))
        >>>     + gr.geom_segment(gr.aes(xend="x_end", yend="y_end", group="level", color="c"))
        >>> )

    """
    ## Check invariants
    # Argument given
    if var is None:
        raise ValueError("No `var` given")
    # Correct number of inputs
    if len(var) != 2:
        raise ValueError("Must provide exactly 2 inputs in `var`.")
    # Inputs available
    var_diff = set(var).difference(set(model.var))
    if len(var_diff) > 0:
        raise ValueError(
            "`var` must be a subset of model.var; missing: {}".format(var_diff)
        )
    # All inputs supported
    var_diff = set(model.var).difference(set(var))
    if len(var_diff) > 0:
        if df is None:
            raise ValueError(
                "Must provide values for remaining model variables using df; " +
                "missing values: {}".format(var_diff)
            )
        var_diff2 = var_diff.difference(set(df.columns))
        if len(var_diff2) > 0:
            raise ValueError(
                "All model variables need values in provided df; " +
                "missing values: {}".format(var_diff2)
            )
    # Finite bound width
    if not all([
            isfinite(model.domain.get_width(v)) and
            (model.domain.get_width(v) > 0)
            for v in var
    ]):
        raise ValueError("All model bounds for `var` must be finite and nonzero")

    # Argument given
    if out is None:
        raise ValueError("No `out` given")
    # Outputs available
    out_diff = set(out).difference(set(model.out))
    if len(out_diff) > 0:
        raise ValueError(
            "`out` must be a subset of model.out; missing: {}".format(out_diff)
        )

    ## Generate data
    xv = linspace(*model.domain.get_bound(var[0]), n_side)
    yv = linspace(*model.domain.get_bound(var[1]), n_side)
    df_x = DataFrame({var[0]: xv})
    df_y = DataFrame({var[1]: yv})
    df_input = (
        df_x
        >> tf_outer(df_outer=df_y)
    )

    # Create singleton level if necessary
    if df is None:
        df = DataFrame({"_foo":[0]})

    ## Loop over provided auxiliary levels
    df_res = DataFrame()
    for i in range(df.shape[0]):
        df_in_tmp = (
            df_input
            >> tf_outer(df_outer=df.iloc[[i]])
        )
        df_out = eval_df(
            model,
            df=df_in_tmp,
        )

        ## Set output threshold levels
        if levels is None:
            levels = dict(zip(
                out,
                [
                    linspace(df_out[o].min(), df_out[o].max(), n_levels + 2)[1:-1]
                    for o in out
                ]
            ))

        ## Run marching squares
        # Output quantity
        for o in out:
            # Reshape data
            Data = reshape(df_out[o].values, (n_side, n_side))
            # Threshold level
            for t in levels[o]:
                segments = marching_square(xv, yv, Data, t)
                df_tmp = DataFrame(
                    data=array(segments).squeeze(),
                    columns=[var[0], var[1], var[0]+"_end", var[1]+"_end"],
                )
                df_tmp["out"] = [o] * df_tmp.shape[0]
                df_tmp["level"] = [t] * df_tmp.shape[0]
                df_tmp = (
                    df_tmp
                    >> tf_outer(df_outer=df.iloc[[i]])
                )

                df_res = concat((df_res, df_tmp), axis=0)

    ## Remove dummy column, if present
    if "_foo" in df_res.columns:
        df_res.drop("_foo", axis=1, inplace=True)

    ## Return the results
    return df_res


ev_contour = add_pipe(eval_contour)
