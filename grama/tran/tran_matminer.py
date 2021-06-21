__all__ = [
    "tran_feat_composition",
    "tf_feat_composition",
]

## Featurize with matminer
try:
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import ElementProperty

except ModuleNotFoundError:
    raise ModuleNotFoundError("module matminer not found")

from grama import add_pipe, pipe
from pandas import concat
from toolz import curry


## Compute matminer featurization
# --------------------------------------------------
@curry
def tran_feat_composition(
    df,
    var_formula="FORMULA",
    preset_name="magpie",
    append=True,
    ignore_errors=True,
    **kwargs,
):
    r"""Featurize a dataset using matminer

    Featurize chemical composition using matminer package.

    Args:
        df (DataFrame): Data to featurize
        var_formula (string): Column in df with chemical formula; formula
            given as string
        append (bool): Append results to original columns?
        preset_name (string): Matminer featurization preset

    Kwargs:
        ignore_errors (bool): Do not throw an error while parsing formulae; set to
            True to return NaN's for invalid formulae.

    Notes:
        - A pre-processor and wrapper for matminer.featurizers.composition

    References:
        Ward, L., Dunn, A., Faghaninia, A., Zimmermann, N. E. R., Bajaj, S., Wang, Q., Montoya, J. H., Chen, J., Bystrom, K., Dylla, M., Chard, K., Asta, M., Persson, K., Snyder, G. J., Foster, I., Jain, A., Matminer: An open source toolkit for materials data mining. Comput. Mater. Sci. 152, 60-69 (2018).

    Examples:
        >>> import grama as gr
        >>> from grama.tran import tf_feat_composition
        >>> (
        >>>     gr.df_make(FORMULA=["C6H12O6"])
        >>>     >> gr.tf_feat_composition()
        >>> )

    """
    ## Check invariants

    ## Featurize
    featurizer = ElementProperty.from_preset(preset_name=preset_name)
    df_res = StrToComposition().featurize_dataframe(
        df[[var_formula]], var_formula, ignore_errors=ignore_errors,
    )
    df_res = featurizer.featurize_dataframe(
        df_res, col_id="composition", ignore_errors=ignore_errors, **kwargs,
    )
    df_res.drop(columns=[var_formula, "composition"], inplace=True)

    ## Concatenate as necessary
    if append:
        df_res = concat((df, df_res), axis=1)

    return df_res


tf_feat_composition = add_pipe(tran_feat_composition)
