## Data Loader: TE-CCA zT Dataset
# Zachary del Rosario (zdelrosario@outlook.com) 2021-03-12
#
from citrination_client import CitrinationClient, PifSystemReturningQuery
from citrination_client import DataQuery, DatasetQuery, Filter
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from pymatgen import Composition

from sl_utils import pifs2df, setResDir

import pandas as pd
import numpy as np
import os
import time

prefix = "zT"
file_responses = prefix + "_responses.csv"
file_features  = prefix + "_features.csv"

## Helper functions
def get_compostion(c):
    """Attempt to parse composition, return None if failed"""

    try:
        return Composition(c)
    except:
        return None

def load_data_zT():
    results_dir = setResDir()

    ## Metadata
    keys_response = [
        'Seebeck coefficient; squared',
        'Electrical resistivity',
        'Thermal conductivity'
       ]
    sign = np.array([
        +1, # Seebeck
        -1, # Electric resistivity
        -1  # Thermal conductivity
       ])

    ## Load data, if possible
    # --------------------------------------------------
    try:
        df_X_all = pd.read_csv(results_dir + file_features)
        X_all = df_X_all.drop(df_X_all.columns[0], axis = 1).values

        df_Y_all = pd.read_csv(results_dir + file_responses)
        Y_all = df_Y_all.drop(df_Y_all.columns[0], axis = 1).values
        print("Cached data loaded.")

    except FileNotFoundError:
        ## Data Import
        # --------------------------------------------------
        # Initialize client
        print("Accessing data from Citrination...")
        site = 'https://citrination.com' # Citrination
        client = CitrinationClient(api_key=os.environ['CITRINATION_API_KEY'], site=site)
        search_client = client.search
        # Aluminum dataset
        dataset_id = 178480 # ucsb_te_roomtemp_seebeck
        system_query = PifSystemReturningQuery(
            size=1000,
            query=DataQuery(
                dataset=DatasetQuery(id=Filter(equal=str(dataset_id)))
            )
           )

        query_result = search_client.pif_search(system_query)
        print("    Found {} PIFs in dataset {}.".format(
            query_result.total_num_hits,
            dataset_id
        ))

        ## Wrangle
        # --------------------------------------------------
        pifs = [x.system for x in query_result.hits]
        # Utility function will tabularize PIFs
        df_response = pifs2df(pifs)
        # Down-select columns to play well with to_numeric
        df_response = df_response[
            ['Seebeck coefficient', 'Electrical resistivity', 'Thermal conductivity']
           ]
        df_response = df_response.apply(pd.to_numeric)

        # Parse chemical compositions
        formulas = [pif.chemical_formula for pif in pifs]

        df_comp = pd.DataFrame(
            columns = ['chemical_formula'],
            data = formulas
           )

        # Join
        df_data = pd.concat([df_comp, df_response], axis = 1)
        print("    Accessed data.")

        # Featurize
        print("Featurizing data...")
        df_data['composition'] = df_data['chemical_formula'].apply(get_compostion)

        f =  MultipleFeaturizer([
            cf.Stoichiometry(),
            cf.ElementProperty.from_preset("magpie"),
            cf.ValenceOrbital(props=['avg']),
            cf.IonProperty(fast=True)
           ])

        X = np.array(f.featurize_many(df_data['composition']))

        # Find valid response values
        keys_original = [
            'Seebeck coefficient',
            'Electrical resistivity',
            'Thermal conductivity'
           ]

        index_valid_response = {
            key: df_data[key].dropna().index.values for key in keys_original
           }

        index_valid_all = df_data[keys_original].dropna().index.values
        X_all           = X[index_valid_all, :]
        Y_all           = df_data[keys_original].iloc[index_valid_all].values

        # Manipulate columns for proper objective values
        Y_all[:, 0] = Y_all[:, 0] ** 2 # Squared seebeck
        print("    Data prepared; {0:} valid observations.".format(X_all.shape[0]))

        # Cache data
        pd.DataFrame(data = X_all).to_csv(results_dir + file_features)
        pd.DataFrame(
            data = Y_all,
            columns = keys_response
        ).to_csv(results_dir + file_responses)
        print("Data cached in results directory.")
        
    return X_all, Y_all, sign, keys_response, prefix

if __name__ == "__main__":
    X_all, Y_all, sign, keys_response, prefix = load_data_zT()
