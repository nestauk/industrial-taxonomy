# Scripts to get processing and processed files

import json
import pickle
import yaml

import pandas as pd

import industrial_taxonomy
project_dir = industrial_taxonomy.project_dir


def get_config(element):
    with open(f"{project_dir}/industrial_taxonomy/model_config.yaml", "r") as infile:
        out = yaml.safe_load(infile)[element]
    return out


def get_table(name):

    return pd.read_csv(f"{project_dir}/data/processed/{name}.csv")


def get_networks():

    with open(f"{project_dir}/data/processed/sector_networks.p", "rb") as infile:
        nets = pickle.load(infile)
    return nets


def get_firm_embeddings():

    return pd.read_csv(f"{project_dir}/data/processed/embeddings_df.csv")


def get_company_sector_lookup():
    with open(
        f"{project_dir}/data/processed/topsbm_cluster_allocations.json", "r"
    ) as infile:
        lookup = json.load(infile)
    return {int(k): v for k, v in lookup.items()}


def get_sector_name_lookup():
    with open(f"{project_dir}/data/processed/topsbm_cluster_names.json", "r") as infile:
        return json.load(infile)

def get_sector_reassignment_outputs():
    with open(f"{project_dir}/data/processed/reassignment_analysis.p", 'rb') as infile:
        return pickle.load(infile)



