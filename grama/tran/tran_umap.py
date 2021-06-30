__all__ = [
    "tran_umap",
    "tf_umap",
]

## Transforms via umap-learn package
try:
    from umap import UMAP
except ModuleNotFoundError:
    raise ModuleNotFoundError("module umap not found")

from grama import add_pipe
from pandas import concat, DataFrame
from toolz import curry


## Compute UMAP
# --------------------------------------------------
@curry
def tran_umap(
    df, var=None, out="xi", keep=True, append=False, n_dim=2, seed=None, **kwargs
):
    r"""UMAP dimension reduction of a dataset

    Apply the UMAP algorithm to reduce the dimensionality of a dataset.

    Args:
        df (DataFrame): Hybrid point results from gr.eval_hybrid()
        var (list or None): Variables in df on which to perform dimension reduction.
            Use None to compute with all variables.
        out (string): Name of reduced-dimensionality output; indexed from 0 .. n_dim-1
        keep (bool): Keep unused columns (outside `var`) in new DataFrame?
        append (bool): Append results to original columns?
        n_dim (int): Target dimensionality

    Kwargs:
        n_neighbors (int): A smaller value emphasizes local structure, larger value emphasizes global structure. Assumed number of nearest-neighbors in clusters. Coenen and Pearce claim this is the most important hyperparameter for UMAP. default=15
        min_dist (float): Minimum distance between mapped points. default=0.1
        metric (str or function): Metric used for distance computations. See url: https://umap-learn.readthedocs.io/en/latest/parameters.html#metric

    Notes:
        - A wrapper for umap.UMAP

    References:
        - McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
        - Andy Coenen, Adam Pearce "Understanding UMAP" url: https://pair-code.github.io/understanding-umap/

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
        data=UMAP(n_components=n_dim, random_state=seed, **kwargs).fit_transform(
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


tf_umap = add_pipe(tran_umap)
