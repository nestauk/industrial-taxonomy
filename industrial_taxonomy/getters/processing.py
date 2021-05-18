# Scripts to get processing and processed files
import pandas as pd
import yaml
import industrial_taxonomy
import pickle

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
