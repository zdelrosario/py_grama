__all__ = [
    "tran_shapley_cohort",
    "tf_shapley_cohort",
]

from grama import pipe
from itertools import chain, combinations
from numpy import all, number, sum, zeros, empty, NaN
from pandas import concat, DataFrame
from scipy.special import comb
from toolz import curry

## Helper
def powerset(iterable):
    s = list(iterable)

    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


## Cohort Shapley
@curry
def tran_shapley_cohort(df, var=None, out=None, bins=20, inds=None):
    """Compute cohort shapley values

    Assess the impact of each variable on selected observations via cohort
    shapley [1]. Shapley values are a game-theoretic way to assess the
    importance of input variables (var) on each of a set of outputs (out). Since
    values are computed on each observation, cohort shapley can distinguish
    cases where a variable has a positive impact on one observation, and a
    negative impact on a different observation.

    Note that cohort shapley is combinatorialy expensive in the number of
    variables, and this expense is multiplied by the number of observations. Use
    with caution in cases of high dimensionality. Consider using the `inds`
    argument to analyze a small subset of your observations.

    Args:
        df (DataFrame): Variable and output data to analyze
        var (list of strings): Input variables
        out (list of strings): Outputs variables
        bins (integer): Number of "bins" to define coordinate refinement distance
        inds (iterable of indices or None): Indices of rows to analyze

    References:
        - [1] Mase, Owen, and Seiler, "Explaining black box decisions by Shapley cohort refinement" (2019) Arxiv

    Examples:
        >>> import grama as gr
        >>> from grama.data import df_stang
        >>> X = gr.Intention()
        >>> # Analyze all observations
        >>> (
        >>>     gr.tran_shapley_cohort(
        >>>         df_stang,
        >>>         var=["thick", "ang"],
        >>>         out=["E"],
        >>>     )
        >>>     >> gr.tf_bind_cols(df_stang)
        >>>     >> gr.tf_filter(X.E_thick < 0)
        >>> )
        >>> # Compute subset of values
        >>> (
        >>>     gr.tran_shapley_cohort(
        >>>         df_stang,
        >>>         var=["thick", "ang"],
        >>>         out=["E"],
        >>>         inds=(
        >>>             df_stang
        >>>             >> gr.tf_filter(X.thick > 0.08)
        >>>         ).index
        >>>     )
        >>>     >> gr.tf_bind_cols(df_stang)
        >>> )

    """
    ## Check invariants
    if not set(var).issubset(set(df.columns)):
        raise ValueError("var must be subset of df.columns")
    if not set(out).issubset(set(df.columns)):
        raise ValueError("out must be subset of df.columns")
    if len(set(var).intersection(set(out))) != 0:
        raise ValueError("var and out must have empty intersection")
    if inds is None:
        inds = range(df.shape[0])

    ## Setup
    s = df.shape[0]  # Number of observations (subjects)
    n = len(var)

    # Determine numeric and categorical columns
    var_numeric = list(df[var].select_dtypes(include=[number]).columns)
    var_cat = list(df[var].drop(columns=var_numeric).columns)

    # Compute distances for coordinate similarity
    df_dist = DataFrame(
        data={col: [(df[col].max() - df[col].min()) / bins] for col in var_numeric}
    )

    # Compute coordinate similarity boolean DataFrame
    df_sim = DataFrame(columns=["_i0", "_i1"] + list(df[var].columns))

    for i in range(s):
        ## Numeric comparison
        df_tmp = (
            df[var_numeric].iloc[i] - df[var_numeric].iloc[(i + 1) :]
        ).abs() <= df_dist[var_numeric].values
        ## Categorical comparison
        df_tmp[var_cat] = df[var_cat].iloc[i] == df[var_cat].iloc[(i + 1) :]
        ## Add subject indices
        df_tmp["_i0"] = [i] * df_tmp.shape[0]
        df_tmp["_i1"] = range((i + 1), s)

        ## Concatenate
        df_sim = concat((df_sim, df_tmp), axis=0, sort=False)

    # Internal functions
    def cohort_indices(t, varset):
        """Build set of cohort indices

        Args:
        t (integer): Target sample index
        varset (iterable): Variables for cohort refinement

        """
        if len(varset) == 0:
            return list(range(s))
        else:
            # Find all pairs similar along given variables
            flags_cohort = all(
                df_sim.drop(columns=["_i0", "_i1"])[[var[i] for i in varset]], axis=1
            ).values

            # Filter to pairs including target t
            df_tmp = df_sim[flags_cohort]
            df_cohort = df_tmp[(df_tmp["_i0"] == t) | (df_tmp["_i1"] == t)]

            # Consolidate index set
            return list(
                set(df_cohort["_i0"]).union(set(df_cohort["_i1"])).union(set((t,)))
            )

    def cohort_mean(t, varset):
        c = cohort_indices(t, varset)
        return df[out].iloc[c].mean().to_frame().T

    def cohort_shapley(j):
        """Cohort shapley for all observations, single variable
        """
        poset = powerset(set(range(n)).difference({j}))
        data = zeros((s, len(out)))
        df_tmp = DataFrame(columns=out, data=data)

        for p in poset:
            den = n * comb(n - 1, len(p))

            for t in range(s):
                if t in inds:
                    t1 = cohort_mean(t, list(set(p).union({j})))
                    t0 = cohort_mean(t, p)

                    df_tmp.iloc[t] = df_tmp.iloc[t] + (t1 - t0).loc[0] / den
                else:
                    df_tmp.iloc[t] = NaN

        return df_tmp

    ## Compute cohort shapley over all variables
    df_res = DataFrame()
    for j in range(n):
        df_tmp = cohort_shapley(j)
        df_tmp.columns = [
            df_tmp.columns[i] + "_" + var[j] for i in range(len(out))
        ]

        df_res = concat((df_res, df_tmp), axis=1)

    return df_res


@pipe
def tf_shapley_cohort(*args, **kwargs):
    return tran_shapley_cohort(*args, **kwargs)
