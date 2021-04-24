from azureml.core import Workspace, Dataset
from datetime import datetime
import numpy as np
import scipy.sparse as sps
import json

def get_credentials():
    with open("credentials.json") as json_data_file:
        data = json.load(json_data_file)
        return data

def get_workspace():
    credentials = get_credentials()
    subscription_id = credentials['subscription_id']
    resource_group = credentials['resource_group']
    workspace_name = credentials['workspace_name']
    return Workspace(subscription_id, resource_group, workspace_name)

def initializing_script(script_name):
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(dt_string, "- Initializing script:", script_name)


def import_data(train_test_split=0.8):
    workspace = get_workspace()

    df = Dataset.get_by_name(workspace, name='movielens_10M')
    df = df.to_pandas_dataframe()

    URM_all = sps.coo_matrix((df["Interaction"].values,  (df["UserID"].values,  df["ItemID"].values)))

    n_interactions = URM_all.nnz
    train_mask = np.random.choice([True,False], n_interactions, p=[train_test_split, 1-train_test_split])
    URM_train = sps.csr_matrix((URM_all.data[train_mask],
                                (URM_all.row[train_mask], URM_all.col[train_mask])))

    test_mask = np.logical_not(train_mask)

    URM_test = sps.csr_matrix((URM_all.data[test_mask],
                                (URM_all.row[test_mask], URM_all.col[test_mask])))
    return URM_train, URM_test
