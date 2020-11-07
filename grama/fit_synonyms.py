__all__ = [
    "fit_nls",
    "ft_nls",
]

## Collection of fitting synonyms: functions implemented in terms of other grama
## verbs

from grama import add_pipe, pipe, custom_formatwarning, Model
from grama import df_make, eval_nls, eval_df, eval_grad_fd, tran_outer
from grama import cp_function, cp_md_det, cp_marginals, cp_copula_gaussian
from toolz import curry
from pandas import concat, DataFrame
from numpy import zeros, diag, atleast_2d, triu_indices
from numpy import sum as npsum
from numpy import power as nppow
from numpy import sqrt as npsqrt
from numpy.linalg import pinv


@curry
def fit_nls(
    df_data, md=None, out=None, verbose=True, uq_method=None, **kwargs,
):
    r"""Fit a model with Nonlinear Least Squares (NLS)

    Estimate best-fit variable levels with nonlinear least squares (NLS), and
    return an executable model with those frozen best-fit levels. Optionally,
    fit a distribution on the parameters to quantify parametric uncertainty.

    Note: This is a *synonym* for eval_nls(); see the documentation for
    eval_nls() for keyword argument options available beyond those listed here.

    Args:
        df_data (DataFrame): Data for estimating best-fit variable levels.
            Variables not found in df_data optimized for fitting.
        md (gr.Model): Model to analyze. All model variables
            selected for fitting must be bounded or random. Deterministic
            variables may have semi-infinite bounds.
        verbose (bool): Print best-fit parameters to console?
        uq_method (str OR None): If string, select method to quantify parameter
            uncertainties. If None, provide best-fit values only. Methods:
            uq_method = "linpool": assume normal errors; linearly approximate
                parameter effects; equally pool variance matrices for each output

    Returns:
        gr.Model: Model for evaluation with best-fit variables frozen to
            optimized levels.

    Examples:
        >>> import grama as gr
        >>> from grama.data import df_trajectory_windowed
        >>> from grama.models import make_trajectory_linear
        >>> X = gr.Intention()
        >>>
        >>> md_trajectory = make_trajectory_linear()
        >>> md_fitted = (
        >>>     df_trajectory_windowed
        >>>     >> gr.ft_nls(
        >>>         md=md_trajectory,
        >>>         uq_method="linpool",
        >>>     )
        >>> )
    """
    ## Check `out` invariants
    if out is None:
        out = md.out
        print("... fit_nls setting out = {}".format(out))

    ## Check invariants
    if md is None:
        raise ValueError("Must provide model md")

    ## Run eval_nls to fit model parameter values
    df_fit = eval_nls(md, df_data=df_data, append=True, **kwargs)
    ## Select best-fit values
    df_best = df_fit.sort_values(by="mse", axis=0).iloc[[0]]
    if verbose:
        print(df_best)

    ## Determine variables to fix
    var_fixed = list(set(md.var).intersection(set(df_best.columns)))
    var_remain = list(set(md.var).difference(set(var_fixed)))

    if len(var_remain) == 0:
        raise ValueError("Resulting model is constant!")

    ## Assemble and return fitted model
    if md.name is None:
        name = "(Fitted Model)"
    else:
        name = md.name + " (Fitted)"

    ## Calibrate parametric uncertainty, if requested
    if uq_method == "linpool":
        ## Precompute data
        df_base = tran_outer(df_data, df_best[var_fixed])
        df_pred = eval_df(md, df=df_base)
        df_grad = eval_grad_fd(md, df_base=df_base, var=var_fixed)

        ## Pool variance matrices
        n_obs = df_data.shape[0]
        n_fixed = len(var_fixed)
        Sigma_pooled = zeros((n_fixed, n_fixed))

        for output in out:
            ## Approximate sigma_sq
            sigma_sq = npsum(
                nppow(df_data[output].values - df_pred[output].values, 2)
            ) / (n_obs - n_fixed)
            ## Approximate (pseudo)-inverse hessian
            var_grad = list(map(lambda v: "D" + output + "_D" + v, var_fixed))
            Z = df_grad[var_grad].values
            Hinv = pinv(Z.T.dot(Z), hermitian=True)

            ## Add variance matrix to pooled Sigma
            Sigma_pooled = Sigma_pooled + sigma_sq * Hinv / n_fixed

        ## Convert to std deviations and correlation
        sigma_comp = npsqrt(diag(Sigma_pooled))
        corr_mat = Sigma_pooled / (atleast_2d(sigma_comp).T.dot(atleast_2d(sigma_comp)))
        corr_data = []
        I, J = triu_indices(n_fixed, k=1)
        for ind in range(len(I)):
            i = I[ind]
            j = J[ind]
            corr_data.append([var_fixed[i], var_fixed[j], corr_mat[i, j]])
        df_corr = DataFrame(data=corr_data, columns=["var1", "var2", "corr"])

        ## Assemble marginals
        marginals = {}
        for ind, var_ in enumerate(var_fixed):
            marginals[var_] = {
                "dist": "norm",
                "loc": df_best[var_].values[0],
                "scale": sigma_comp[ind],
            }

        ## Construct model with Gaussian copula
        md_res = (
            Model(name)
            >> cp_md_det(md=md)
            >> cp_marginals(**marginals)
            >> cp_copula_gaussian(df_corr=df_corr)
        )

    ## Return deterministic model
    elif uq_method is None:
        md_res = (
            Model(name)
            >> cp_function(
                lambda x: df_best[var_fixed].values,
                var=var_remain,
                out=var_fixed,
                name="Fix variable levels",
            )
            >> cp_md_det(md=md)
        )

    else:
        raise ValueError("uq_method option {} not recognized".format(uq_method))

    return md_res


ft_nls = add_pipe(fit_nls)
