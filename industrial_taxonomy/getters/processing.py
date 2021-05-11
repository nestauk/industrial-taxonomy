# Scripts to get processing and processed files

import yaml
import industrial_taxonomy

project_dir = industrial_taxonomy.project_dir


def get_config(element):
    with open(f"{project_dir}/industrial_taxonomy/model_config.yaml", "r") as infile:
        out = yaml.safe_load(infile)[element]
    return out
