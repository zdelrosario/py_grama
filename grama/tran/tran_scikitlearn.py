__all__ = [
    "tran_tsne",
    "tf_tsne",
]

## Transforms via sklearn package
try:
    from sklearn.manifold import TSNE
except ModuleNotFoundError:
    raise ModuleNotFoundError("module sklearn not found")

from grama import pipe
from toolz import curry
from pandas import concat, DataFrame

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
        df_res = concat((df_res, df[var_leftover]), axis=1)
    if append:
        df_res = concat((df_res, df[var]), axis=1)

    return df_res


@pipe
def tf_tsne(*args, **kwargs):
    return tran_tsne(*args, **kwargs)
