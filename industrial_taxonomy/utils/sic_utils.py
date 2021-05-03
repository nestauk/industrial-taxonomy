# Create sic4 code - name lookup

import requests
import re
import pandas as pd
import industrial_taxonomy
import os
import logging

project_dir = industrial_taxonomy.project_dir

_SIC_OUTPUT_FILE = f"{project_dir}/data/raw/sic_2007.xls"
_SIC_TAXONOMY_URL = "https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/ukstandardindustrialclassificationofeconomicactivities/uksic2007/sic2007summaryofstructurtcm6.xls"


def save_sic_taxonomy():
    """Fetch SIC taxonomy and save as excel file"""
    if os.path.exists(_SIC_OUTPUT_FILE) is False:
        logging.info("Fetching SIC taxonomy")
        response = requests.get(_SIC_TAXONOMY_URL)
        with open(_SIC_OUTPUT_FILE, "wb") as f:
            f.write(response.content)
    else:
        logging.info("Already fetched SIC taxonomy")


def load_sic_taxonomy():
    """Load SIC taxonomy into a dataframe"""
    return pd.read_excel(
        _SIC_OUTPUT_FILE,
        skiprows=1,
        dtype={"Division": str, "Group": str, "Class": str, "Sub Class": str},
    )


def extract_sic_code_description(var_name):
    """Extracts codes and descriptions from SIC table
    Args:
        var_name (str): level of SIC we want to extract a lookup for
    Returns:
        A lookup between the variable codes and their description
    """
    table = load_sic_taxonomy()

    loc = list(table.columns).index(var_name)  # Find the location of class
    select = table.iloc[:, [loc, loc + 1]].dropna()  # Extract variable & description
    select.columns = [var_name, "description"]

    select[var_name] = [re.sub(r"\.", "", str(x)) for x in select[var_name]]
    return select.set_index(var_name)["description"].to_dict()
