__all__ = [
    "tran_tsne",
    "tf_tsne",
    "tran_poly",
    "tf_poly",
]

## Transforms via sklearn package
try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import PolynomialFeatures
except ModuleNotFoundError:
    raise ModuleNotFoundError("module sklearn not found")

from grama import add_pipe
from pandas import concat, DataFrame
from toolz import curry


## Compute t-SNE
# --------------------------------------------------
@curry
def tran_tsne(
    df, var=None, out="xi", keep=True, append=False, n_dim=2, seed=None, **kwargs
):
    r"""t-SNE dimension reduction of a dataset

    Apply the t-SNE algorithm to reduce the dimensionality of a dataset.

    Args:
        df (DataFrame): Hybrid point results from gr.eval_hybrid()
        var (list or None): Variables in df on which to perform dimension reduction.
            Use None to compute with all variables.
        out (string): Name of reduced-dimensionality output; indexed from 0 .. n_dim-1
        keep (bool): Keep unused columns (outside `var`) in new DataFrame?
        append (bool): Append results to original columns?
        n_dim (int): Target dimensionality

    Kwargs:
        n_iter (int): Maximum number of iterations for optimization. As Wattenberg et al. note, this is the most important parameter in using t-SNE. If you see strange "pinched" shapes, increase n_iter.
        perplexity (int): Usually between 5 and 50. Low perplexity means local variations dominate; High perplexity tends to merge clusters.
        early_exaggeration (float):
        learning_rate (float):

    Notes:
        - A wrapper for sklearn.manifold.TSNE

    References:
        Scikit-learn: Machine Learning in Python, Pedregosa et al. JMLR 12, pp. 2825-2830, 2011.

        Wattenberg, Viegas, and Johnson, "How to use t-SNE effectively" (2016) Distil.pub

    Examples:

    """
    ## Check invariants
    if var is None:
        var = list(df.columns).copy()
    else:
        var = list(var).copy()
        diff = set(var).difference(set(df.columns))
        if len(diff) > 0:
            raise ValueError(
                "`var` must be subset of `df.columns`\n" "diff = {}".format(diff)
            )
    var_leftover = list(set(df.columns).difference(set(var)))

    ## Reduce dimensionality
    df_res = DataFrame(
        data=TSNE(n_components=n_dim, random_state=seed, **kwargs).fit_transform(
            df[var].values
        ),
        columns=[out + "{}".format(i) for i in range(n_dim)],
    )

    ## Concatenate as necessary
    if keep:
        df_res = concat(
            (df_res.reset_index(drop=True), df[var_leftover].reset_index(drop=True)),
            axis=1,
        )
    if append:
        df_res = concat(
            (df_res.reset_index(drop=True), df[var].reset_index(drop=True)), axis=1
        )

    return df_res


tf_tsne = add_pipe(tran_tsne)

# --------------------------------------------------
@curry
def tran_poly(df, degree=None, var=None, keep=True, **kwargs):
    r"""Compute polynomial features of a dataset

    Compute polynomial features of a dataset.

    Args:
        df (DataFrame): Hybrid point results from gr.eval_hybrid()

    Kwargs:
        degree (int): Maximum degree of polynomial features
        var (list or None): Variables in df on which to perform dimension reduction.
            Use None to compute with all variables.
        keep (bool): Keep unused columns (outside `var`) in new DataFrame?
        interaction_only (bool): If true, only produce interaction features
        include_bias (bool): If true, include a constant feature term (bias)

    Notes:
        - A wrapper for sklearn.preprocessing.PolynomialFeatures

    References:
        Scikit-learn: Machine Learning in Python, Pedregosa et al. JMLR 12, pp. 2825-2830, 2011.

    Examples:

    """
    ## Check invariants
    if var is None:
        var = list(df.columns).copy()
    else:
        var = list(var).copy()
        diff = set(var).difference(set(df.columns))
        if len(diff) > 0:
            raise ValueError(
                "`var` must be subset of `df.columns`\n" "diff = {}".format(diff)
            )
    var_leftover = list(set(df.columns).difference(set(var)))

    ## Compute the features
    fit = PolynomialFeatures(degree)
    X_feat = fit.fit_transform(df[var].values)
    var_feat = fit.get_feature_names(var)

    ## Package results
    df_feat = DataFrame(data=X_feat, columns=var_feat)

    if keep:
        return concat((df_feat, df[var_leftover].reset_index(drop=True)), axis=1)

    return df_feat


tf_poly = add_pipe(tran_poly)
