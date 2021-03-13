import numpy as np
import os
import pandas as pd
import pickle

from functools import reduce
from pypif_sdk.readview import ReadView

## Metaprogramming
##################################################
# Append a suffix to a function's __name__
class _nameSuffix(object):
    def __init__(self, suffix):
        self.suffix = suffix
    def __call__(self, f):
        f.__name__ = f.__name__ + self.suffix
        return f

## Parsing
##################################################
# Set results directory
def setResDir(env_var = "SL_RESULTS"):
    """Set the results directory

    :param env_var: Environment variable to reference
    :param type: string
    :returns: path to results directory
    :rtype: string

    """
    results_dir = os.environ[env_var]
    assert os.path.isdir(results_dir), "Results directory {} must exist!".format(results_dir)
    # Ensure trailing slash
    if results_dir[-1] != "/":
        results_dir = results_dir + "/"

    return results_dir

# Get a PIF scalar
def parsePifKey(pif, key):
    """Parse a single pif key for single scalar values; return nan if no scalar found.

    :param pif: PIF to access
    :type pif: pif
    :param key: key to access data
    :type key: string
    :returns: scalar value or np.nan
    :rtype:

    """
    if (key in ReadView(pif).keys()):
        if 'scalars' in dir(ReadView(pif)[key]):
            try:
                return ReadView(pif)[key].scalars[0].value
            except IndexError:
                return np.nan
        else:
            return np.nan
    else:
        return np.nan

# Flatten a collection of PIFs
def pifs2df(pifs):
    """Converts a collection of PIFs to tabular data
    Very simple, purpose-built utility script. Converts an iterable of PIFs
    to a dataframe. Returns the superset of all PIF keys as the set of columns.
    Non-scalar values are converted to nan.

    Usage
        df = pifs2df(pifs)
    Arguments
        pifs = an iterable of PIFs
    Returns
        df = Pandas DataFrame

    examples
        import os
        from citrination_client import CitrinationClient
        from citrination_client import PifSystemReturningQuery, DatasetQuery
        from citrination_client import DataQuery, Filter

        ## Set-up citrination search client
        site = "https://citrination.com"
        client = CitrinationClient(api_key = os.environ["CITRINATION_API_KEY"], site = site)
        search_client = client.search

        ## Query the Agrawal (2014) dataset
        system_query = \
            PifSystemReturningQuery(
                size = 500,
                query = DataQuery(
                    dataset = DatasetQuery(id = Filter(equal = "150670"))
                )
            )
        query_result = search_client.pif_search(system_query)
        pifs = [x.system for x in query_results.hits]

        ## Rectangularize the pifs
        df = pifs2df(pifs)
    """
    ## Consolidate superset of keys
    key_sets = [set(ReadView(pif).keys()) for pif in pifs]
    keys_ref = reduce(
        lambda s1, s2: s1.union(s2),
        key_sets
    )

    ## Rectangularize
    ## TODO: Append dataframes, rather than using a comprehension
    df_data = \
        pd.DataFrame(
            columns = keys_ref,
            data = [
                [
                    parsePifKey(pif, key) \
                    for key in keys_ref
                ] for pif in pifs
            ]
        )

    return df_data
