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

from grama import pipe
from toolz import curry
from pandas import concat

## Compute matminer featurization
# --------------------------------------------------
@curry
def tran_feat_composition(
        df,
        var_formula="FORMULA",
        preset_name="magpie",
        append=True,
):
    r"""Featurize a dataset using matminer

    Featurize chemical composition using matminer package.

    Args:
        df (DataFrame): Data to featurize
        var_formula (string): Column in df with chemical formula; formula
            given as string
        append (bool): Append results to original columns?
        preset_name (string): Matminer featurization preset

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
    df_res = StrToComposition().featurize_dataframe(df[[var_formula]], var_formula)
    df_res = featurizer.featurize_dataframe(df_res, col_id="composition")
    df_res.drop(columns=[var_formula, "composition"], inplace=True)

    ## Concatenate as necessary
    if append:
        df_res = concat((df, df_res), axis=1)

    return df_res


@pipe
def tf_feat_composition(*args, **kwargs):
    return tran_feat_composition(*args, **kwargs)
