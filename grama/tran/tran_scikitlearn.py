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
    df,
    var=None,
    out="xi",
    keep=True,
    append=False,
    n_dim=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate=200.0,
    n_iter=1000,
    n_iter_without_progress=300,
    min_grad_norm=1e-07,
    metric="euclidean",
    init="random",
    verbose=0,
    random_state=None,
    method="barnes_hut",
    angle=0.5,
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

    Notes:
        - A wrapper for sklearn.manifold.TSNE

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

    ## Reduce dimensionality
    df_res = DataFrame(
        data=TSNE(
            n_components=n_dim,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
            n_iter_without_progress=n_iter_without_progress,
            min_grad_norm=min_grad_norm,
            metric=metric,
            init="random",
            verbose=verbose,
            random_state=random_state,
            method=method,
            angle=angle,
        ).fit_transform(df[var].values),
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
