# Get UNSPSC data
import pandas as pd
import industrial_taxonomy
import re

project_dir = industrial_taxonomy.project_dir


def get_unspsc() -> pd.DataFrame:
    """Reads the UNSPCC data"""
    unsspc = "UNSPSC English v230701.xlsx"
    f = pd.read_excel(f"{project_dir}/data/raw/{unsspc}", skiprows=9)
    f.columns = [re.sub(" ", "_", x.lower()) for x in f.columns]
    return f
